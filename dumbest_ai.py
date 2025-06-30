#This is Dumbest AI !!! Do not Argue. :D
import os
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib
import pickle
import shutil # Dosya işlemleri için
import time # Gecikme için eklendi
import uuid # EKLENDİ: uuid modülü import edildi

import gradio as gr
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

from langchain.text_splitter import RecursiveCharacterTextSplitter, SpacyTextSplitter
from langchain.chains import LLMChain # LLMChain import edildi
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_ollama import OllamaLLM
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.embeddings import Embeddings

# ParentDocumentRetriever için yeni importlar
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore # Basit bir doküman deposu

from sentence_transformers import SentenceTransformer, CrossEncoder
from convert_encoding import convert_to_utf8_no_bom

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# NLTK data download
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.warning("NLTK 'punkt' or 'stopwords' not found. Attempting to download...")
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        logger.info("NLTK 'punkt' and 'stopwords' downloaded successfully.")
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {e}. Some features might be affected.")


# Configuration
@dataclass
class RAGConfig:
    vector_store_dir: str = "vectorstore"
    documents_dir: str = "documents"
    chat_history_file: str = "chat_history.json"
    cache_dir: str = "cache"
    
    # Chunking Ayarları (Small-to-Big için)
    parent_chunk_size: int = 1024 # Daha büyük ana (parent) parçalar
    parent_chunk_overlap: int = 256
    child_chunk_size: int = 64  # **Daha küçük varsayılan değer: Vektörleştirmek için daha küçük (child) parçalar**
    child_chunk_overlap: int = 16 # **Daha küçük varsayılan değer**

    top_k_retrieval: int = 10
    top_k_rerank: int = 5
    temperature: float = 0.1
    max_memory_length: int = 10
    confidence_threshold: float = 0.7
    embedding_batch_size: int = 64 # Embedding batch boyutu
    # ÖNEMLİ: vectorstore_add_batch_size, ChromaDB'nin tek seferde işleyebileceği child chunk sayısı olmalı (yaklaşık 5000-5461)
    vectorstore_add_batch_size: int = 4000 # Varsayılan olarak 4000 olarak güncellendi.
    # Processed file metadata'yı tutacak dosya
    processed_files_meta_file: str = os.path.join("cache", "processed_files_meta.json")


config = RAGConfig()

# CUDA Setup
print("🚀 CUDA Test")
print("✅ CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("🧠 Device:", torch.cuda.get_device_name(torch.cuda.current_device()))
    device = "cuda"
else:
    print("❌ CUDA not available, CPU kullanılacak.")
    device = "cpu"
torch.cuda.empty_cache()

# RAG-Fusion (Reciprocal Rank Fusion - RRF) fonksiyonu
def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[Document, float]]], k: int = 60
) -> List[Tuple[Document, float]]:
    """
    Reciprocal Rank Fusion (RRF) kullanarak birden fazla sıralı listeyi birleştirir.
    Daha düşük rank daha iyidir. İlk eleman rank 1.
    """
    fused_scores = {}
    k_rrf = 60.0 # RRF için sabir k değeri (genellikle 60)

    # Her bir sıralı liste üzerinde dön
    for ranked_list in ranked_lists:
        # Check if ranked_list is empty to avoid errors with enumerate
        if not ranked_list:
            continue
        for rank, (doc, original_score) in enumerate(ranked_list):
            # Benzersiz ID'yi metadatadan al veya içerikten oluştur
            doc_id = doc.metadata.get('id')
            if doc_id is None:
                doc_id = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
                doc.metadata['id'] = doc_id # ID'yi belgeye ekle
            
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {'doc': doc, 'score': 0.0}
            
            # RRF formülü: 1 / (k_rrf + rank)
            fused_scores[doc_id]['score'] += 1.0 / (k_rrf + rank + 1) # Rank 0'dan başladığı için +1

    # Fused skorlara göre sırala
    sorted_results = sorted(fused_scores.values(), key=lambda x: x['score'], reverse=True)
    
    # Orjinal Document objelerini ve füzyon skorlarını döndür
    return [(item['doc'], item['score']) for item in sorted_results[:k]]


class AdvancedEmbeddings(Embeddings):
    """Gelişmiş embedding sınıfı - cache ve batch processing ile"""

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1"): # nomic-ai modeli varsayılan
        self.model_name = model_name
        self.cache_file = os.path.join(config.cache_dir, "embeddings_cache.pkl")
        self.cache = self._load_cache()

        try:
            self.model = SentenceTransformer(
                model_name,
                device=device,
                trust_remote_code=True
            )
            logger.info(f"✅ Embedding model loaded on {device}")
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}. Fallback to CPU.")
            self.model = SentenceTransformer(
                model_name,
                device="cpu",
                trust_remote_code=True
            )
            logger.info("✅ Fallback to CPU mode.")

    def _load_cache(self) -> Dict:
        os.makedirs(config.cache_dir, exist_ok=True)
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e: # Catch specific exception
                logger.warning(f"❌ Embedding cache could not be loaded from {self.cache_file} ({e}). Starting with empty cache.")
                return {}
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.error(f"❌ Failed to save embedding cache: {e}")

    def _get_cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest() # encoding belirtildi

    def embed_documents(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        # config.embedding_batch_size'ı kullan
        if batch_size is None:
            batch_size = config.embedding_batch_size

        if not texts: 
            logger.warning("AdvancedEmbeddings.embed_documents received an empty list of texts. Returning empty embeddings.")
            return []

        embeddings = []
        uncached_texts = []
        uncached_indices = []

        # Cache kontrolü
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text) 
            if cache_key in self.cache:
                embeddings.append(self.cache[cache_key])
            else:
                embeddings.append(None) # Yer tutucu
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Yeni embeddingleri oluştur
        if uncached_texts:
            logger.info(f"🔄 Embedding {len(uncached_texts)} new documents (in batches of {batch_size})...")
            new_embeddings_list = [] # Batch'lerden gelen embedding'leri toplamak için
            
            # tqdm ile ilerleme çubuğu ekleyelim
            from tqdm import tqdm
            # İşlemciye uygun sayıda worker kullanma (Windows'da sorunlu olabilir, None bırakılabilir)
            num_workers = 0 # Default to 0 for Windows compatibility, or os.cpu_count() - 1 for Linux/macOS
            
            for i in tqdm(range(0, len(uncached_texts), batch_size), desc="Embedding Batches"):
                batch = uncached_texts[i:i+batch_size]
                batch_embeddings = self.model.encode(
                    batch,
                    show_progress_bar=False, # tqdm kendi progress barını kullanacak
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    device=self.model.device, # Modelin çalıştığı cihazı belirt
                    num_workers=num_workers # Batch yükleme için worker sayısı
                )
                new_embeddings_list.extend(batch_embeddings)

            # Cache'e kaydet ve orijinal listeye yerleştir
            for i, text_idx in enumerate(uncached_indices):
                text = uncached_texts[i]
                embedding_data = new_embeddings_list[i].tolist() # Numpy array'i listeye dönüştür
                cache_key = self._get_cache_key(text)
                self.cache[cache_key] = embedding_data
                embeddings[text_idx] = embedding_data

            self._save_cache()
        else:
            logger.info("✅ All embeddings found in cache, skipping embedding generation.")

        # None değerlerini gerçek embeddingler ile değiştir (placeholder kullanıyorsanız)
        final_embeddings = []
        for emb in embeddings:
            if emb is None:
                logger.error("Hata: Bir embedding boş kaldı, bu olmamalıydı. Boş string embedding'i oluşturuluyor.")
                # Bu durum olmamalıydı ama olursa boş bir embedding ile devam et
                final_embeddings.append(self.model.encode("", convert_to_numpy=True, normalize_embeddings=True, device=self.model.device).tolist())
            else:
                final_embeddings.append(emb)

        # Final check before returning: If input texts were not empty but embeddings came out empty
        if not final_embeddings and texts: 
             logger.error("Hata: Girdi metinleri boş olmamasına rağmen AdvancedEmbeddings boş embedding listesi döndürdü. Bu, modelin veya girdinin sorunlu olduğunu gösterir.")
        
        return final_embeddings

    def embed_query(self, text: str) -> List[float]:
        cache_key = self._get_cache_key(text)
        if cache_key in self.cache:
            return self.cache[cache_key]

        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True, device=self.model.device)
        self.cache[cache_key] = embedding.tolist()
        self._save_cache()
        return embedding.tolist()

class SemanticChunker:
    """Semantik olarak anlamlı parçalara bölen sınıf"""

    def __init__(self, embeddings: AdvancedEmbeddings, threshold: float = 0.75):
        self.embeddings = embeddings
        self.threshold = threshold

    def split_text(self, text: str, metadata: Dict = None) -> List[Document]:
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return [Document(page_content=text, metadata=metadata or {})]

        # Cümle embeddingleri (burada kendi embedding modelini kullanmalı)
        sentence_embeddings = self.embeddings.model.encode(sentences, convert_to_numpy=True, device=self.embeddings.model.device)

        # Benzerlik hesapla
        chunks = []
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            # Kosinüs benzerliği hesaplamak için normalize edilmiş vektörler önemlidir
            similarity = cosine_similarity(
                [sentence_embeddings[i-1]],
                [sentence_embeddings[i]]
            )[0][0]

            if similarity > self.threshold:
                current_chunk.append(sentences[i])
            else:
                # Yeni chunk başlat
                chunk_text = " ".join(current_chunk)
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata={**(metadata or {}), "chunk_type": "semantic"}
                ))
                current_chunk = [sentences[i]]

        # Son chunk'ı ekle
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(Document(
                page_content=chunk_text,
                metadata={**(metadata or {}), "chunk_type": "semantic"}
            ))

        return chunks

class HybridRetriever:
    """Hibrit arama: Semantic + Keyword-based"""

    def __init__(self, parent_retriever: ParentDocumentRetriever, bm25_documents: List[Document]):
        self.parent_retriever = parent_retriever # ParentDocumentRetriever instance'ı
        self.bm25_documents = bm25_documents # BM25 için tüm child dokümanları
        # Self.embeddings'i kaldırdık çünkü reranker direkt olarak modelini init'te alıyor.
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device) # Reranker'ı da GPU'ya taşı
        
        # Stop words: Başlangıçta boş bir set ile güvenli başlatma
        self.stop_words = set() 
        try:
            # Türkçe ve İngilizce stop word'leri birleştirme
            nltk_stop_words = set(stopwords.words('english'))
            nltk_stop_words.update(stopwords.words('turkish'))
            self.stop_words = nltk_stop_words
        except LookupError:
            logger.warning("NLTK stopwords could not be loaded. Running without stopwords.")
        
        # BM25 indeksi oluştur
        self.bm25 = self._build_bm25_index()


    def _build_bm25_index(self):
        """BM25 indeksi oluştur"""
        # Sadece BM25 için belge içeriklerini tokenize et
        corpus = []
        for doc in self.bm25_documents: # BM25 için kullanılan doküman listesi
            tokens = word_tokenize(doc.page_content.lower())
            tokens = [t for t in tokens if t.isalnum() and t not in self.stop_words] 
            corpus.append(tokens)

        if not corpus:
            logger.warning("BM25 index cannot be built: No valid document content found for tokenization.")
            return None
        return BM25Okapi(corpus)

    def _preprocess_query(self, query: str) -> str:
        """Query'yi zenginleştir"""
        expanded_terms = []
        tokens = word_tokenize(query.lower())

        for token in tokens:
            if token.isalnum() and token not in self.stop_words:
                expanded_terms.append(token)
                if token in ["nasıl", "how"]:
                    expanded_terms.extend(["yöntem", "method", "way"])
                elif token in ["nedir", "what"]:
                    expanded_terms.extend(["tanım", "definition", "açıklama"])
        return " ".join(expanded_terms)

    def retrieve(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Hibrit retrieval (RRF ile)"""
        processed_query = self._preprocess_query(query)

        # 1. Semantic search (ParentDocumentRetriever'ın underlying vectorstore'dan child chunks al)
        # ParentDocumentRetriever'ın child retriever'ını kullanarak skorlu child chunk'ları al
        semantic_child_results = self.parent_retriever.vectorstore.similarity_search_with_score(
            processed_query, k=k*3 # RRF ve reranking için daha fazla al
        )
        
        # 2. BM25 search (child dokümanlar üzerinde)
        bm25_child_results = []
        if self.bm25 is not None:
            query_tokens = word_tokenize(processed_query.lower())
            query_tokens = [t for t in query_tokens if t.isalnum() and t not in self.stop_words]
            if query_tokens:
                bm25_scores = self.bm25.get_scores(query_tokens)
                # BM25 sonuçlarını (document, score) formatına dönüştür
                # Her dokümanın BM25 skoru ile eşleştirilmesi gerekir
                temp_bm25_results = []
                for i, score in enumerate(bm25_scores):
                    if i < len(self.bm25_documents):
                        temp_bm25_results.append((self.bm25_documents[i], float(score)))
                
                temp_bm25_results.sort(key=lambda x: x[1], reverse=True) # Skorlara göre sırala
                bm25_child_results = temp_bm25_results[:k*3] # RRF için daha fazla al
            else:
                logger.warning("BM25 search skipped: Preprocessed query has no valid tokens.")
        else:
            logger.warning("BM25 index is not available. Performing only semantic search.")

        # RRF kullanarak sonuçları birleştir
        # Her iki liste de (Document, score) formatında olmalı ve Document.metadata['id'] içermeli.
        fused_child_results = reciprocal_rank_fusion([semantic_child_results, bm25_child_results], k=k) # top_k kadar sonuç al
        
        return fused_child_results


    def rerank_results(self, query: str, results: List[Tuple[Document, float]], top_k: int = 5) -> List[Tuple[Document, float]]:
        """Cross-encoder ile reranking"""
        if len(results) <= top_k or not results:
            return results

        # Reranking için query-document pairs hazırla
        # context için sadece page_content alıyoruz.
        pairs = [(query, doc.page_content) for doc, _ in results]

        try:
            rerank_scores = self.reranker.predict(pairs)

            reranked_results = [
                (results[i][0], float(score)) # Orijinal doküman objesini kullan
                for i, score in enumerate(rerank_scores)
            ]
            reranked_results.sort(key=lambda x: x[1], reverse=True)

            return reranked_results[:top_k]
        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Returning original top-k results.")
            return results[:top_k]

class AdvancedRAGSystem:
    """Gelişmiş RAG sistemi"""

    def __init__(self):
        self.embeddings = AdvancedEmbeddings() # Initialized here
        self.docstore = InMemoryStore() # ParentDocumentRetriever için doküman deposu
        self.vectorstore = None # Child chunks için Chroma
        self.parent_document_retriever = None # Langchain ParentDocumentRetriever
        self.hybrid_retriever = None # Kendi HybridRetriever'ımız
        self.documents_for_bm25 = [] # BM25 için kullanılan tüm child dokümanları (metadata içerir)

        self.llm = None
        self.memory = None
        self.qa_chain = None # Şimdi bir LLMChain olacak
        self.response_cache = {}
        self.processed_files_meta = self._load_processed_files_meta()

    def _get_file_hash(self, file_path: str) -> str:
        """Dosyanın MD5 hash'ini hesaplar."""
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"❌ Dosya hash'i alınamadı {file_path}: {e}")
            return ""

    def _load_processed_files_meta(self) -> Dict[str, Dict]:
        """İşlenmiş dosyaların metadata'sını yükler."""
        if os.path.exists(config.processed_files_meta_file):
            try:
                with open(config.processed_files_meta_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"❌ Processed files metadata could not be loaded: {e}. Starting fresh.")
        return {}

    def _save_processed_files_meta(self):
        """İşlenmiş dosyaların metadata'sını kaydeder."""
        os.makedirs(os.path.dirname(config.processed_files_meta_file), exist_ok=True)
        try:
            with open(config.processed_files_meta_file, 'w', encoding='utf-8') as f:
                json.dump(self.processed_files_meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"❌ Failed to save processed files metadata: {e}")

    def build_knowledge_base(self, force_rebuild: bool = False):
        """Gelişmiş knowledge base oluştur veya artımlı olarak güncelle"""
        os.makedirs(config.documents_dir, exist_ok=True)
        os.makedirs(config.vector_store_dir, exist_ok=True)
        os.makedirs(config.cache_dir, exist_ok=True) # Cache dizini de oluşmalı

        if force_rebuild:
            logger.info("🔄 Force rebuild requested. Deleting existing vectorstore, cache, and metadata.")
            
            # Explicitly clear references to potentially locked resources
            self.vectorstore = None
            self.parent_document_retriever = None
            self.hybrid_retriever = None
            # Önemli: self.embeddings'i yeniden başlatmak, eski önbellek dosyasının bırakılmasına yardımcı olur.
            self.embeddings = AdvancedEmbeddings() 
            
            # Allow a small delay for OS to release file handles (optional, but can help)
            time.sleep(0.1) 

            if os.path.exists(config.vector_store_dir):
                try:
                    shutil.rmtree(config.vector_store_dir)
                    logger.info(f"✅ Deleted existing vectorstore directory: {config.vector_store_dir}")
                except OSError as e:
                    logger.error(f"❌ Could not delete vectorstore directory '{config.vector_store_dir}': {e}. Please ensure no other process is using these files and try again.")
                    raise # Re-raise the error to stop the process if deletion fails

            if os.path.exists(config.cache_dir):
                try:
                    shutil.rmtree(config.cache_dir)
                    logger.info(f"✅ Deleted existing cache directory: {config.cache_dir}")
                except OSError as e:
                    logger.error(f"❌ Could not delete cache directory '{config.cache_dir}': {e}. Please ensure no other process is using these files and try again.")
                    # Don't re-raise for cache if vectorstore deletion already worked or is main concern
            
            self.processed_files_meta = {}
            self._save_processed_files_meta()
            self.docstore = InMemoryStore() # Yeni bir InMemoryStore instance'ı oluştur

        # Chroma'yı yükle veya boş oluştur (child chunks için)
        # Chroma'nın dizini yoksa hata verebilir, ensure_directory_exists ile kontrol edelim.
        if not os.path.exists(config.vector_store_dir):
            os.makedirs(config.vector_store_dir)

        self.vectorstore = Chroma(
            persist_directory=config.vector_store_dir,
            embedding_function=self.embeddings
        )

        # ParentDocumentRetriever'ı initialize et
        # Bu Retriever'ı artık sadece child_splitter ve parent_splitter'ı kullanmak için tutuyoruz.
        # add_documents metodunu artık doğrudan çağırmıyoruz, çünkü manuel olarak yöneteceğiz.
        self.parent_document_retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore, # Placeholder, aslında manuel olarak ekleyeceğiz
            docstore=self.docstore,
            child_splitter=RecursiveCharacterTextSplitter(chunk_size=config.child_chunk_size, chunk_overlap=config.child_chunk_overlap),
            parent_splitter=RecursiveCharacterTextSplitter(chunk_size=config.parent_chunk_size, chunk_overlap=config.parent_chunk_overlap),
        )


        logger.info("🔍 Checking for document changes...")
        convert_to_utf8_no_bom(config.documents_dir)

        current_files_in_dir = set()
        new_or_modified_original_documents_to_add = [] 
        
        # 1. Mevcut belgeleri tara ve değişiklikleri belirle
        for root, _, filenames in os.walk(config.documents_dir):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                relative_file_path = os.path.relpath(file_path, config.documents_dir)
                current_files_in_dir.add(relative_file_path)

                current_hash = self._get_file_hash(file_path)
                last_processed_meta = self.processed_files_meta.get(relative_file_path)

                if not last_processed_meta or last_processed_meta['hash'] != current_hash:
                    logger.info(f"🆕/📝 Found new or modified document: {relative_file_path}")
                    
                    original_docs_from_file = self._load_raw_document(file_path)
                    if original_docs_from_file:
                        logger.info(f"Successfully loaded {len(original_docs_from_file)} raw document(s) from {relative_file_path}.")
                        for doc in original_docs_from_file:
                            doc.metadata['id'] = hashlib.md5(relative_file_path.encode('utf-8')).hexdigest()
                            doc.metadata['original_file_path'] = relative_file_path
                            new_or_modified_original_documents_to_add.append(doc)
                    else:
                        logger.warning(f"⚠️ Could not load original document: {relative_file_path}. It might be empty, unreadable, or an unsupported format.")
                    
                    self.processed_files_meta[relative_file_path] = {
                        'hash': current_hash,
                        'last_processed': datetime.now().isoformat(),
                    }
                else:
                    pass # Already processed and not modified, no action needed for original documents


        # 2. Silinen belgeleri belirle ve docstore/vectorstore'dan kaldır
        deleted_file_paths = []
        for processed_file_path, meta in list(self.processed_files_meta.items()):
            if processed_file_path not in current_files_in_dir:
                logger.info(f"🗑️ Found deleted document (metadata will be removed): {processed_file_path}")
                parent_id_to_delete = hashlib.md5(processed_file_path.encode('utf-8')).hexdigest()
                
                # Docstore'dan kaldır
                if hasattr(self.docstore, 'delete') and parent_id_to_delete in self.docstore.docs:
                    self.docstore.delete([parent_id_to_delete])
                    logger.info(f"Deleted parent document {parent_id_to_delete} from docstore.")
                
                # Vectorstore'dan ilgili child chunk'ları kaldır
                # Bu, ChromaDB'nin delete metodunu kullanarak metadata filtrelemesiyle yapılabilir.
                try:
                    self.vectorstore._collection.delete(where={"parent_id": parent_id_to_delete})
                    logger.info(f"Deleted child chunks for parent_id {parent_id_to_delete} from vectorstore.")
                except Exception as e:
                    logger.error(f"❌ Error deleting child chunks from vectorstore for {parent_id_to_delete}: {e}")

                deleted_file_paths.append(processed_file_path)
                del self.processed_files_meta[processed_file_path]
        
        if deleted_file_paths:
            logger.warning(f"Note: {len(deleted_file_paths)} deleted files processed.")

        # 3. Yeni/Değiştirilmiş orijinal belgeleri manuel olarak işleyip Chroma'ya ekle
        if new_or_modified_original_documents_to_add:
            total_original_docs_to_add = len(new_or_modified_original_documents_to_add)
            logger.info(f"➕ Processing {total_original_docs_to_add} new/modified original documents for vectorstore.")

            all_new_child_chunks = []
            
            # Parent dokümanları parçala ve docstore'a ekle, çocuk chunk'ları topla
            for i, original_doc in enumerate(new_or_modified_original_documents_to_add):
                logger.info(f"    Chunking original document {i+1}/{total_original_docs_to_add}: {original_doc.metadata.get('original_file_path', 'N/A')}")
                
                # Parent dokümanı parent_splitter ile parçala
                parent_chunks_from_original_doc = self.parent_document_retriever.parent_splitter.split_documents([original_doc])
                
                for parent_piece_of_original_doc in parent_chunks_from_original_doc:
                    # Yeni bir ID atayalım, çünkü bu aslında bir "alt-parent" parça oluyor orijinal belge için
                    # Bu ID, ParentDocumentRetriever'ın internal parent_id'si gibi davranacak
                    current_parent_piece_id = str(uuid.uuid4()) # Her parent parçasına yeni bir ID
                    parent_piece_of_original_doc.metadata['id'] = current_parent_piece_id # Metadata'ya ekle
                    parent_piece_of_original_doc.metadata['original_file_path'] = original_doc.metadata['original_file_path']

                    # Düzeltme: mset metodu key-value tuple'larından oluşan bir liste bekler
                    self.docstore.mset([(current_parent_piece_id, parent_piece_of_original_doc)]) 
                    
                    # Bu parent parçayı child_splitter ile parçala
                    child_chunks_from_parent_piece = self.parent_document_retriever.child_splitter.split_documents([parent_piece_of_original_doc])
                    
                    for child_chunk in child_chunks_from_parent_piece:
                        # Child chunk'a orijinal dosya yolunu ve parent ID'yi ekle
                        child_chunk.metadata['original_file_path'] = original_doc.metadata['original_file_path']
                        child_chunk.metadata['parent_id'] = current_parent_piece_id # Bu child'ın ait olduğu parent parçanın ID'si
                        child_chunk.metadata['parent_doc_id'] = current_parent_piece_id # Langchain'in beklediği anahtar
                        all_new_child_chunks.append(child_chunk)
            
            # Şimdi tüm toplanan child chunk'ları küçük partiler halinde Chroma'ya ekle
            if all_new_child_chunks:
                logger.info(f"Total {len(all_new_child_chunks)} child chunks generated. Adding to vectorstore in batches of {config.vectorstore_add_batch_size}.")
                for i in range(0, len(all_new_child_chunks), config.vectorstore_add_batch_size):
                    sub_batch = all_new_child_chunks[i:i + config.vectorstore_add_batch_size]
                    
                    try:
                        self.vectorstore.add_documents(sub_batch)
                        logger.info(f"    Added batch {int(i/config.vectorstore_add_batch_size) + 1}/{(len(all_new_child_chunks) + config.vectorstore_add_batch_size - 1) // config.vectorstore_add_batch_size} to ChromaDB.")
                    except Exception as e:
                        logger.error(f"❌ Error adding child chunks batch to ChromaDB: {e}. Batch size was {len(sub_batch)}")
                        raise

            logger.info("✅ Vectorstore updated with new/modified documents.")
        else:
            logger.info("ℹ️ No new or modified original documents to add to vectorstore (all are up-to-date or no files found).")
        
        # BM25 için güncel documents_for_bm25 listesini doldur (tüm child chunks)
        # Bu kısım, tüm child chunk'ları yeniden yükleyerek güncel listeyi oluşturur.
        # Bu, önceki ekleme/silme işlemlerinden sonra tutarlılığı sağlar.
        self.documents_for_bm25 = []
        # Tüm işlenmiş dosyaların metadataları üzerinden geçerek child chunk'ları yeniden yükle
        # Bu, `processed_files_meta`'da olan tüm dosyaların BM25 için yeniden yüklenmesini sağlar.
        for file_path_rel, meta in list(self.processed_files_meta.items()): # list() çağrısı, döngü sırasında dict değişirse sorun olmaması için
            full_file_path = os.path.join(config.documents_dir, file_path_rel)
            # _load_single_document_and_chunk zaten child chunk'lar üretip döndürüyor.
            loaded_chunks_for_bm25 = self._load_single_document_and_chunk(full_file_path)
            if loaded_chunks_for_bm25:
                # logger.info(f"Loaded {len(loaded_chunks_for_bm25)} child chunks for BM25 from {file_path_rel}.") # Çok fazla log olabilir
                self.documents_for_bm25.extend(loaded_chunks_for_bm25)
            else:
                logger.warning(f"⚠️ Could not re-load chunks for BM25 from {full_file_path}. Document might be missing, unreadable, or produced no chunks after splitting.")


        if not self.documents_for_bm25:
            logger.warning("No documents loaded into self.documents_for_bm25 for BM25 retriever, keyword search may not function correctly.")

        # HybridRetriever'ı yeniden initialize et
        self.hybrid_retriever = HybridRetriever(self.parent_document_retriever, self.documents_for_bm25)

        self._save_processed_files_meta()

        logger.info(f"✅ Knowledge base updated. Total {len(self.documents_for_bm25)} child chunks for BM25.")
        return self.vectorstore

    def _load_raw_document(self, file_path: str) -> List[Document]:
        """Tek bir orijinal belgeyi parçalamadan yükler."""
        try:
            ext = os.path.splitext(file_path)[1].lower()
            loader = None
            
            if ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif ext in ['.txt', '.md', '.py', '.cs', '.json', '.xml', '.html', '.css', '.js', '.ts', '.tsx', '.jsx', '.go', '.java', '.php', '.rb', '.swift', '.kt', '.c', '.cpp', '.h', '.hpp', '.scss', '.cshtml']:
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                logger.warning(f"⚠️ Unsupported file type for loading: '{ext}' in {file_path}. Skipping.")
                return []

            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents from {file_path} using {type(loader).__name__}.")
            
            if not documents:
                logger.warning(f"Loader returned an empty document list for {file_path}. File might be empty, corrupted, or unreadable.")
                return []

            valid_documents = []
            for doc in documents:
                if doc.page_content.strip():
                    valid_documents.append(doc)
                    logger.info(f"Document from {file_path} (page {doc.metadata.get('page', 'N/A')}): content length = {len(doc.page_content)} characters.")
                else:
                    logger.warning(f"Document from {file_path} (page {doc.metadata.get('page', 'N/A')}) has empty or whitespace-only content after loading. Skipping.")

            if not valid_documents:
                logger.warning(f"All documents loaded from {file_path} were empty or contained only whitespace. Returning empty list.")
                return []

            for doc in valid_documents:
                doc.metadata.update({
                    'file_type': ext,
                    'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                    'created_at': datetime.now().isoformat(),
                    'source': file_path
                })
            return valid_documents
        except Exception as e:
            logger.warning(f"⚠️ Hata: Orijinal belge yüklenemedi {file_path}: {e}")
            return []

    def _load_single_document_and_chunk(self, file_path: str) -> List[Document]:
        """
        Tek bir belgeyi yükler ve child chunk olarak parçalar.
        """
        try:
            raw_docs = self._load_raw_document(file_path)
            if not raw_docs:
                logger.warning(f"No raw documents to chunk for BM25 from {file_path}. Skipping chunking for this file.")
                return []

            child_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.child_chunk_size,
                chunk_overlap=config.child_chunk_overlap
            )
            chunks = child_splitter.split_documents(raw_docs)

            if not chunks:
                logger.warning(f"Child splitter produced no chunks for {file_path} with chunk_size {config.child_chunk_size}. Document content might be too small or contains unchunkable content.")
                return []

            file_base_hash = hashlib.md5(os.path.relpath(file_path, config.documents_dir).encode('utf-8')).hexdigest()
            
            processed_chunks = [] 
            for i, chunk in enumerate(chunks):
                if chunk.page_content.strip():
                    chunk.metadata['id'] = f"{file_base_hash}_{i}"
                    chunk.metadata['original_file_path'] = os.path.relpath(file_path, config.documents_dir)
                    # Buradaki parent_id, orijinal dosyanın ID'si olmalı
                    chunk.metadata['parent_id'] = file_base_hash 
                    processed_chunks.append(chunk)
                else:
                    logger.warning(f"Skipping empty or whitespace-only chunk generated from {file_path} (original chunk index: {i}).")

            if not processed_chunks:
                logger.warning(f"All generated chunks for {file_path} were empty or whitespace-only after content check. Returning empty list.")
            return processed_chunks
        except Exception as e:
            logger.warning(f"⚠️ Hata: Belge child chunk olarak parçalanamadı {file_path}: {e}")
            return []

    def initialize_llm_chain(self):
        """LLM chain'i initialize et"""
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=config.max_memory_length
        )

        self.llm = OllamaLLM(
            model="llama3",
            temperature=config.temperature,
            top_p=0.9,
            repeat_penalty=1.1
        )

        prompt_template = """Sen uzman bir yapay zeka asistansın ve özellikle .NET ve web geliştirme konularında deneyimlisin. Görevin, aşağıda verilen 'BAĞLAM BİLGİLERİ'ne dayanarak kullanıcının sorularını doğru, eksiksiz, detaylı ve açıklayıcı bir şekilde yanıtlamaktır.

BAĞLAM BİLGİLERİ:
{context}

SOHBET GEÇMİŞİ:
{chat_history}

KULLERANIICI SORUSU: {question}

YÖNERGE:
1. Yanıtını **öncelikle ve ağırlıklı olarak** 'BAĞLAM BİLGİLERİ' içinde yer alan bilgilere dayanarak oluştur. Kendi genel bilgilerini kullanmaktan kaçın, ancak bağlamı daha iyi açıklamak veya yapılandırmak için yardımcı olabilirsin.
2. Eğer 'BAĞLAM BİLGİLERİ'nde soruyu yanıtlamak için yeterli veya ilgili bilgi yoksa, 'Üzgünüm, bu bilgiyi sağlanan belgelerde yeterince detaylı bulamadım veya ilgili bir bilgiye rastlamadım. Başka bir şey sormak ister misiniz?' şeklinde nazikçe belirt ve **asla yanıt UYDURMA**.
3. Cevabını her zaman 'BAĞLAM BİLGİLERİ'ndeki ilgili kaynak numaralarıyla destekle (örn: [Kaynak 1], [Kaynak 2]).
4. Yanıtlarını organize et (maddeler, başlıklar kullanabilirsin) ve karmaşık konuları anlaşılır bir dille açıkla. Mümkünse örnekler veya senaryolar sun.
5. Yanıtının sonunda, cevabının ne kadar güvenilir olduğunu 0 ile 1 arasında bir ondalık sayı olarak 'Güvenilirlik Skoru:' şeklinde belirt. (Örn: Güvenilirlik Skoru: 0.95)

YANITIM:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "chat_history", "question"]
        )

        self.qa_chain = LLMChain(llm=self.llm, prompt=prompt, verbose=True)


    def enhanced_query_processing(self, query: str) -> Dict[str, Any]:
        """Gelişmiş query işleme"""
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
        if query_hash in self.response_cache:
            logger.info("📋 Using cached response for query.")
            return {**self.response_cache[query_hash], "cached": True}
        
        if self.hybrid_retriever is None:
            return {
                "answer": "Üzgünüm, bilgi bankası henüz hazır değil veya boş.",
                "sources": [],
                "confidence": 0.0,
                "error": "Retriever is not initialized."
            }

        # Hibrit retrieval (BM25 ve Semantic arama ve RRF birleştirme)
        retrieved_child_docs_with_scores = self.hybrid_retriever.retrieve(
            query, k=config.top_k_retrieval * 2
        )
        
        # Reranking (hala child dokümanlar üzerinde)
        reranked_child_docs_with_scores = self.hybrid_retriever.rerank_results(
            query, retrieved_child_docs_with_scores, top_k=config.top_k_rerank
        )
        
        # Şimdi Small-to-Big adımını uyguluyoruz:
        final_context_docs = []
        unique_parent_ids = set()
        
        for child_doc, score in reranked_child_docs_with_scores:
            parent_id = child_doc.metadata.get('parent_doc_id') # Langchain'in atadığı parent_doc_id
            if not parent_id: # Fallback: Bizim manuel atadığımız parent_id
                parent_id = child_doc.metadata.get('parent_id')

            if parent_id and parent_id not in unique_parent_ids:
                parent_docs_from_store = self.docstore.mget([parent_id]) 
                parent_doc = parent_docs_from_store[0] if parent_docs_from_store and parent_docs_from_store[0] else None

                if parent_doc:
                    parent_doc.metadata['original_file_path'] = child_doc.metadata.get('original_file_path', 'N/A')
                    final_context_docs.append((parent_doc, score))
                    unique_parent_ids.add(parent_id)
                else:
                    logger.warning(f"Parent document with ID {parent_id} not found in docstore for child {child_doc.metadata.get('id', 'N/A')}. Using child document as fallback.")
                    final_context_docs.append((child_doc, score))
            else:
                if parent_id is None:
                     final_context_docs.append((child_doc, score))
                elif parent_id in unique_parent_ids:
                    pass
                else:
                    final_context_docs.append((child_doc, score))

        if not final_context_docs and reranked_child_docs_with_scores:
            logger.warning("No parent documents retrieved. Falling back to reranked child documents for context.")
            final_context_docs = reranked_child_docs_with_scores

        context = "\n\n".join([
            f"[Kaynak {i+1}] {doc.page_content}"
            for i, (doc, score) in enumerate(final_context_docs)
        ])
        
        try:
            chat_history_messages = self.memory.load_memory_variables({})["chat_history"]
            
            prompt_inputs = {
                "question": query,
                "chat_history": chat_history_messages,
                "context": context
            }
            
            result = self.qa_chain.invoke(prompt_inputs)
            answer = result["text"]
            
            self.memory.save_context({"input": query}, {"output": answer})
            
            confidence = self._calculate_confidence(query, final_context_docs, answer)
            
            response = {
                "answer": answer,
                "sources": [
                    {
                        "id": doc.metadata.get('id', 'N/A'),
                        "original_file_path": doc.metadata.get('original_file_path', doc.metadata.get('source', 'N/A')),
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata,
                        "score": float(score)
                    }
                    for doc, score in final_context_docs
                ],
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }
            
            self.response_cache[query_hash] = response
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Query processing failed: {e}")
            return {
                "answer": "Üzgünüm, sorgunuzu işlerken bir hata oluştu.",
                "sources": [],
                "confidence": 0.0,
                "error": str(e)
            }

    def _calculate_confidence(self, query: str, docs: List[Tuple[Document, float]], answer: str) -> float:
        """Basit confidence skoru hesapla"""
        if not docs:
            return 0.0
        
        avg_retrieval_score = np.mean([score for _, score in docs])
        
        answer_length_factor = min(1.0, len(answer.split()) / 50.0)
        answer_length_factor = max(0.3, answer_length_factor)

        if avg_retrieval_score < 0.2:
            confidence = 0.1
        else:
            confidence = (avg_retrieval_score * 0.7) + (answer_length_factor * 0.3)
        
        if "bulamadım" in answer.lower() or "sağlanan belgeler" in answer.lower() or "üzgünüm" in answer.lower():
            confidence = min(confidence, 0.2)

        return min(1.0, max(0.0, confidence))

    def get_follow_up_suggestions(self, query: str, response: Dict[str, Any]) -> List[str]:
        """Follow-up soru önerileri"""
        suggestions = []
        
        query_lower = query.lower()
        
        if "nedir" in query_lower or "what is" in query_lower:
            suggestions.append(f"Bu konuda daha detaylı bilgi verir misin?")
            suggestions.append(f"Örneklerle açıklar mısın?")
        
        if "nasıl" in query_lower or "how" in query_lower:
            suggestions.append(f"Alternatif yöntemler nelerdir?")
            suggestions.append(f"Adım adım anlatır mısın?")
        
        if "neden" in query_lower or "why" in query_lower:
            suggestions.append(f"Bu durumun sonuçları nelerdir?")
            suggestions.append(f"Başka sebepleri var mı?")
        
        sources = response.get("sources", [])
        if sources:
            source_files = list(set([os.path.basename(s.get('original_file_path', '')) for s in sources if s.get('original_file_path')]))
            if source_files:
                suggestions.append(f"{source_files[0]} dosyasında başka hangi bilgiler var?")
                if len(source_files) > 1:
                    suggestions.append(f"{source_files[1]} dosyasının içeriğini özetler misin?")

        return suggestions[:3]

def save_chat_history(history: List[Dict]):
    """Chat history kaydet"""
    try:
        with open(config.chat_history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save chat history: {e}")

def load_chat_history() -> List[Dict]:
    """Chat history yükle"""
    if os.path.exists(config.chat_history_file):
        try:
            with open(config.chat_history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load chat history: {e}")
            return []
    return []

def create_gradio_interface():
    """Gelişmiş Gradio arayüzü"""

    rag_system = AdvancedRAGSystem()

    logger.info("🏗️ Building knowledge base...")
    vectorstore = rag_system.build_knowledge_base()

    if vectorstore is None:
        logger.error("❌ Knowledge base creation failed!")
        return None

    logger.info("🔗 Initializing LLM chain...")
    rag_system.initialize_llm_chain()

    chat_history = load_chat_history()

    with gr.Blocks(
        title="🧠 Gelişmiş RAG Sistemi",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .message-wrap {
            padding: 15px !important;
        }
        """
    ) as demo:

        gr.Markdown("""
        # 🧠 Gelişmiş RAG (Retrieval-Augmented Generation) Sistemi

        **Özellikler:**
        - 🔍 Hibrit Arama (Semantic + Keyword)
        - 🎯 Akıllı Reranking (Cross-Encoder)
        - 💾 Yanıt Cache'i
        - 📊 Güvenilirlik Skoru
        - 📚 Kaynak Gösterimi
        - 💡 Follow-up Önerileri
        - 🚀 Artımlı Bilgi Bankası Güncelleme
        - ✨ **Small-to-Big Chunking**
        - 🔥 **RAG-Fusion (Reciprocal Rank Fusion - RRF)**
        """)

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    value=chat_history,
                    type="messages",
                    height=500,
                    label="Sohbet Geçmişi"
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Sorunuzu buraya yazın ve Enter'a basın...",
                        lines=3,
                        label="Mesajınız",
                        scale=4
                    )
                    submit_btn = gr.Button("Gönder", variant="primary", scale=1)

                suggestions = gr.Radio(
                    choices=[],
                    label="💡 Önerilen Sorular",
                    visible=False
                )

            with gr.Column(scale=1):
                system_status = gr.HTML(
                    value=f"""
                    <div style="padding: 15px; border-radius: 10px; background: #f0f8ff;">
                        <h3>📊 Sistem Durumu</h3>
                        <p><strong>Device:</strong> {device.upper()}</p>
                        <p><strong>BM25 Parçaları:</strong> {len(rag_system.documents_for_bm25) if rag_system.documents_for_bm25 else 0}</p>
                        <p><strong>Model:</strong> Llama3</p>
                        <p><strong>Cache:</strong> Aktif</p>
                        <p><strong>Retrieval:</strong> Hibrit (RRF)</p>
                        <p><strong>Chunking:</strong> Small-to-Big</p>
                    </div>
                    """
                )

                response_details = gr.JSON(
                    label="📋 Son Yanıt Detayları",
                    visible=True
                )

                performance_metrics = gr.HTML(
                    value="<div style='padding: 10px;'><h4>⚡ Performans</h4><p>Hazır...</p></div>"
                )
        
        kb_status = gr.HTML(value="<p>Bilgi bankası hazır.</p>")

        with gr.Accordion("🔧 Sistem Ayarları", open=False):
            with gr.Row():
                parent_chunk_size_slider = gr.Slider(
                    minimum=128, maximum=2048, value=config.parent_chunk_size,
                    label="Parent Chunk Size", info="Ana belge parçalama boyutu (LLM'e verilen bağlam)"
                )
                parent_chunk_overlap_slider = gr.Slider(
                    minimum=0, maximum=512, value=config.parent_chunk_overlap,
                    label="Parent Chunk Overlap", info="Ana belge parçaları arası çakışma"
                )
                child_chunk_size_slider = gr.Slider(
                    minimum=16, maximum=512, value=config.child_chunk_size,
                    label="Child Chunk Size", info="Alt belge parçalama boyutu (Embedding ve arama için)"
                )
                child_chunk_overlap_slider = gr.Slider(
                    minimum=0, maximum=128, value=config.child_chunk_overlap,
                    label="Child Chunk Overlap", info="Alt belge parçaları arası çakışma"
                )
            
            with gr.Row():
                top_k_retrieval_slider = gr.Slider(
                    minimum=1, maximum=30, value=config.top_k_retrieval,
                    label="Top-K Retrieval", info="RRF öncesi kaç belge getirilsin"
                )
                top_k_rerank_slider = gr.Slider(
                    minimum=1, maximum=15, value=config.top_k_rerank,
                    label="Top-K Rerank", info="Rerank sonrası kaç belge LLM'e gönderilsin"
                )
                temperature_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, value=config.temperature,
                    label="Temperature", info="LLM yaratıcılık seviyesi"
                )
                embedding_batch_size_slider = gr.Slider(
                    minimum=1, maximum=256, value=config.embedding_batch_size,
                    label="Embedding Batch Size", info="Embedding işlemi için aynı anda işlenen parça sayısı"
                )
                vectorstore_add_batch_size_slider = gr.Slider(
                    minimum=1, maximum=5000, value=config.vectorstore_add_batch_size,
                    label="Vectorstore Add Batch Size", info="Vektör veritabanına aynı anda eklenecek belge (child chunk) sayısı"
                )


            save_settings_btn = gr.Button("⚙️ Ayarları Kaydet ve Uygula", variant="secondary")

            def save_settings(parent_chunk_size, parent_chunk_overlap, child_chunk_size, child_chunk_overlap, top_k_retrieval, top_k_rerank, temperature, embedding_batch_size, vectorstore_add_batch_size):
                config.parent_chunk_size = int(parent_chunk_size)
                config.parent_chunk_overlap = int(parent_chunk_overlap)
                config.child_chunk_size = int(child_chunk_size)
                config.child_chunk_overlap = int(child_chunk_overlap)
                config.top_k_retrieval = int(top_k_retrieval)
                config.top_k_rerank = int(top_k_rerank)
                config.temperature = float(temperature)
                config.embedding_batch_size = int(embedding_batch_size)
                config.vectorstore_add_batch_size = int(vectorstore_add_batch_size)

                rag_system.embeddings.batch_size = config.embedding_batch_size
                
                logger.info("⚙️ Ayarlar güncellendi. Bilgi bankası artımlı olarak güncelleniyor...")
                try:
                    rag_system.build_knowledge_base(force_rebuild=False)
                    rag_system.initialize_llm_chain()
                    return f"<p style='color: green;'>✅ Ayarlar kaydedildi ve bilgi bankası güncellendi!</p>"
                except Exception as e:
                    logger.error(f"❌ Ayarları uygularken veya bilgi bankasını güncellerken hata: {e}")
                    return f"<p style='color: red;'>❌ Hata: Ayarlar uygulanamadı! {str(e)}</p>"


            save_settings_btn.click(
                fn=save_settings,
                inputs=[
                    parent_chunk_size_slider, parent_chunk_overlap_slider,
                    child_chunk_size_slider, child_chunk_overlap_slider,
                    top_k_retrieval_slider, top_k_rerank_slider,
                    temperature_slider, embedding_batch_size_slider,
                    vectorstore_add_batch_size_slider
                ],
                outputs=kb_status
            )


        with gr.Accordion("📚 Bilgi Bankası Yönetimi", open=False):
            rebuild_btn = gr.Button("🔄 Bilgi Bankasını Yeniden Oluştur (Tamamen)", variant="stop")
            update_btn = gr.Button("⬆️ Değişiklikleri Kontrol Et ve Güncelle", variant="primary")

            def rebuild_knowledge_base_full():
                logger.info("Full rebuild initiated.")
                try:
                    rag_system.build_knowledge_base(force_rebuild=True)
                    rag_system.initialize_llm_chain()
                    return f"<p style='color: green;'>✅ Bilgi bankası tamamen yeniden oluşturuldu! ({len(rag_system.documents_for_bm25)} child belge parçası)</p>"
                except Exception as e:
                    logger.error(f"❌ Full rebuild failed: {e}")
                    return f"<p style='color: red;'>❌ Hata: {str(e)}</p>"

            def update_knowledge_base_incremental():
                logger.info("Incremental update initiated.")
                try:
                    rag_system.build_knowledge_base(force_rebuild=False)
                    rag_system.initialize_llm_chain()
                    return f"<p style='color: green;'>✅ Bilgi bankası artımlı olarak güncellendi! ({len(rag_system.documents_for_bm25)} güncel child belge parçası)</p>"
                except Exception as e:
                    logger.error(f"❌ Incremental update failed: {e}")
                    return f"<p style='color: red;'>❌ Hata: {str(e)}</p>"


            rebuild_btn.click(
                fn=rebuild_knowledge_base_full,
                outputs=kb_status
            )
            update_btn.click(
                fn=update_knowledge_base_incremental,
                outputs=kb_status
            )


    return demo

def main():
    """Ana fonksiyon"""
    try:
        logger.info("🚀 Starting Advanced RAG System...")

        demo = create_gradio_interface()

        if demo is None:
            logger.error("❌ Failed to create Gradio interface!")
            return

        logger.info("🌐 Starting Gradio server...")
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False,
            show_error=True,
            inbrowser=True
        )

    except KeyboardInterrupt:
        logger.info("🛑 Sistem kullanıcı tarafından durduruldu")
    except Exception as e:
        logger.error(f"❌ Sistem hatası: {e}")
        raise

if __name__ == "__main__":
    main()
