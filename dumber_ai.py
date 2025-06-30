#This is Dumber AI !!! Do not Argue. lol
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
import uuid # uuid modülü import edildi

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

from tqdm import tqdm # İlerleme çubuğu için eklendi

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


# Configuration (cachemain_.py'den alınan ve birleştirilen Config)
@dataclass
class RAGConfig:
    # Genel Dizin Ayarları
    vector_store_dir: str = "vectorstore" # ChromaDB'nin saklanacağı dizin
    documents_dir: str = "documents" # Kullanıcının belgelerinin yükleneceği dizin
    chat_history_file: str = "chat_history.json" # Sohbet geçmişi dosyası
    cache_dir: str = "cache" # Embedding cache ve diğer önbelleklerin saklanacağı dizin
    processed_files_meta_file: str = os.path.join("cache", "processed_files_meta.json") # İşlenen dosya meta verilerini tutacak dosya

    # LLM Ayarları
    llm_model: str = "llama3" # Ollama model adı
    ollama_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.5 # LLM sıcaklığı

    # Embedding ve Reranker Modelleri
    embeddings_model_name: str = "nomic-ai/nomic-embed-text-v1" # Embedding modeli
    cross_encoder_model_name: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2" # Cross-encoder reranker modeli

    # Chunking Ayarları (Small-to-Big için)
    parent_chunk_size: int = 1024 # Daha büyük ana (parent) parçalar
    parent_chunk_overlap: int = 256 # Parent parçalar arası çakışma
    child_chunk_size: int = 64  # Vektörleştirmek için daha küçük (child) parçalar
    child_chunk_overlap: int = 16 # Child parçalar arası çakışma

    # Retrieval ve Rerank Ayarları
    bm25_top_k: int = 5 # BM25 ile alınacak top-k sonuç sayısı
    vector_top_k: int = 5 # Vektör araması ile alınacak top-k sonuç sayısı
    top_k_retrieval: int = 30 # Hibrit arama sonrası (RRF öncesi) alınacak toplam belge sayısı
    top_k_rerank: int = 10 # Reranking sonrası alınacak nihai belge sayısı

    # Diğer Ayarlar
    max_memory_length: int = 10 # Sohbet geçmişinde tutulacak konuşma sayısı
    confidence_threshold: float = 0.7 # Güven eşiği (şu an kullanılmıyor ama gelecekte eklenebilir)
    embedding_batch_size: int = 64 # Embedding batch boyutu
    vectorstore_add_batch_size: int = 4000 # ChromaDB'ye tek seferde eklenecek child chunk sayısı (performans için)

    # Takip Soruları Ayarları
    suggestion_template: str = """
    Aşağıdaki sohbet geçmişi ve yanıta göre, kullanıcıya sorulabilecek 3 adet kısa, alakalı ve çeşitli takip sorusu oluştur.
    Sorular, kullanıcının daha fazla bilgi edinmesine yardımcı olmalı veya konuyu daha derinlemesine keşfetmelidir.
    Her bir soruyu yeni bir satırda liste olarak formatla (örn: - Soru 1).
    Aynı veya çok benzer soruları tekrarlama.
    Eğer bağlamda önerilecek bir şey yoksa boş liste döndür.

    Sohbet Geçmişi:
    {chat_history}

    Yanıt: {answer}

    Olası Takip Soruları:
    """
    suggestion_llm_model: str = "llama3" # Takip soruları için farklı bir model kullanılabilir
    suggestion_temperature: float = 0.5

    rag_template: str = """
    Sen bir yapay zeka asistanısın. Sağlanan bağlamı kullanarak soruları yanıtla.
    Yalnızca sağlanan bağlamdan bilgi kullan. Yanıtlarınızı kısa ve öz tutun.
    Bağlamda cevap yoksa, "Üzgünüm, bu konu hakkında yeterli bilgiye sahip değilim. Lütfen başka bir soru sorun veya bilgi bankasını güncelleyin." deyin.
    Kesinlikle uydurma bilgi verme.
    Soruya bağlı olarak takip soruları önermeyi unutma.

    Konuşma Geçmişi:
    {chat_history}

    Bağlam:
    {context}

    Soru: {question}
    Yanıt:
    """

config = RAGConfig()

# CUDA Setup
logger.info("🚀 CUDA Test")
if torch.cuda.is_available():
    logger.info("✅ CUDA available.")
    logger.info(f"🧠 Device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    device = "cuda"
else:
    logger.info("❌ CUDA not available, CPU kullanılacak.")
    device = "cpu"
torch.cuda.empty_cache()

# Gerekli dizinleri oluştur
os.makedirs(config.documents_dir, exist_ok=True)
os.makedirs(config.vector_store_dir, exist_ok=True)
os.makedirs(config.cache_dir, exist_ok=True)


# RAG-Fusion (Reciprocal Rank Fusion - RRF) fonksiyonu
def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[Document, float]]], k: int = 60
) -> List[Tuple[Document, float]]:
    """
    Reciprocal Rank Fusion (RRF) kullanarak birden fazla sıralı listeyi birleştirir.
    Daha düşük rank daha iyidir. İlk eleman rank 1.
    """
    fused_scores = {}
    k_rrf = 60.0 # RRF için sabit k değeri (genellikle 60)

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

    def __init__(self, model_name: str = config.embeddings_model_name):
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
            except Exception as e:
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
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def embed_documents(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        if batch_size is None:
            batch_size = config.embedding_batch_size

        if not texts: 
            logger.warning("AdvancedEmbeddings.embed_documents received an empty list of texts. Returning empty embeddings.")
            return []

        embeddings = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text) 
            if cache_key in self.cache:
                embeddings.append(self.cache[cache_key])
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            logger.info(f"🔄 Embedding {len(uncached_texts)} new documents (in batches of {batch_size})...")
            new_embeddings_list = []
            
            num_workers = 0 # Default to 0 for Windows compatibility, or os.cpu_count() - 1 for Linux/macOS
            
            for i in tqdm(range(0, len(uncached_texts), batch_size), desc="Embedding Batches"):
                batch = uncached_texts[i:i+batch_size]
                batch_embeddings = self.model.encode(
                    batch,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    device=self.model.device,
                    num_workers=num_workers
                )
                new_embeddings_list.extend(batch_embeddings)

            for i, text_idx in enumerate(uncached_indices):
                text = uncached_texts[i]
                embedding_data = new_embeddings_list[i].tolist()
                cache_key = self._get_cache_key(text)
                self.cache[cache_key] = embedding_data
                embeddings[text_idx] = embedding_data

            self._save_cache()
        else:
            logger.info("✅ All embeddings found in cache, skipping embedding generation.")

        final_embeddings = []
        for emb in embeddings:
            if emb is None:
                logger.error("Hata: Bir embedding boş kaldı, bu olmamalıydı. Boş string embedding'i oluşturuluyor.")
                final_embeddings.append(self.model.encode("", convert_to_numpy=True, normalize_embeddings=True, device=self.model.device).tolist())
            else:
                final_embeddings.append(emb)

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

        sentence_embeddings = self.embeddings.model.encode(sentences, convert_to_numpy=True, device=self.embeddings.model.device)

        chunks = []
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            similarity = cosine_similarity(
                [sentence_embeddings[i-1]],
                [sentence_embeddings[i]]
            )[0][0]

            if similarity > self.threshold:
                current_chunk.append(sentences[i])
            else:
                chunk_text = " ".join(current_chunk)
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata={**(metadata or {}), "chunk_type": "semantic"}
                ))
                current_chunk = [sentences[i]]

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(Document(
                page_content=chunk_text,
                metadata={**(metadata or {}), "chunk_type": "semantic"}
            ))

        return chunks


class HybridRetriever:
    """Hibrit arama: Semantic + Keyword-based (BM25) + Reranking + RRF"""

    def __init__(self, parent_retriever: ParentDocumentRetriever, bm25_documents: List[Document], embeddings: AdvancedEmbeddings):
        self.parent_retriever = parent_retriever # ParentDocumentRetriever instance'ı
        self.bm25_documents = bm25_documents # BM25 için tüm child dokümanları
        self.embeddings = embeddings # Embedding sınıfı
        self.reranker = CrossEncoder(config.cross_encoder_model_name, device=device) # Reranker'ı da GPU'ya taşı
        
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
        for doc in self.bm25_documents:
            # BM25 için kullanılan doküman listesi
            tokens = word_tokenize(doc.page_content.lower())
            tokens = [t for t in tokens if t.isalnum() and t not in self.stop_words]
            corpus.append(tokens)
        
        if not corpus:
            logger.warning("BM25 index cannot be built: No valid document content found for tokenization.")
            return None
        return BM25Okapi(corpus)

    def _preprocess_query(self, query: str) -> str:
        """Query'yi zenginleştir (isteğe bağlı)"""
        expanded_terms = []
        tokens = word_tokenize(query.lower())
        for token in tokens:
            if token.isalnum() and token not in self.stop_words:
                expanded_terms.append(token)
            # if token in ["nasıl", "how"]:
            #     expanded_terms.extend(["yöntem", "method", "way"])
            # elif token in ["nedir", "what"]:
            #     expanded_terms.extend(["tanım", "definition", "açıklama"])
        return " ".join(expanded_terms)

    def retrieve(self, query: str, k: int = config.top_k_retrieval) -> List[Document]:
        """Hibrit retrieval (RRF ve Reranking ile)"""
        processed_query = self._preprocess_query(query)

        # 1. Semantic search (ParentDocumentRetriever'ın underlying vectorstore'dan child chunks al)
        # ParentDocumentRetriever'ın child retriever'ını kullanarak skorlu child chunk'ları al
        # semantic_child_results = self.parent_retriever.vectorstore.similarity_search_with_score(
        #     processed_query, k=k*3 # RRF ve reranking için daha fazla al
        # )
        # Langchain'in ParentDocumentRetriever'ı doğrudan retrieve metoduyla parent belgeleri döndürür.
        # Bizim burada child belgeleri skorlarıyla almamız gerekiyor.
        # Bu, Chroma'nın search metodunu kullanarak yapılabilir.
        semantic_child_results_with_scores = self.parent_retriever.vectorstore.similarity_search_with_score(
            processed_query, k=config.vector_top_k * 3 # Vektör arama top-k'sının 3 katını al
        )
        # Doküman ve skor tuple listesini hazırla
        semantic_results_for_rrf = [(doc, score) for doc, score in semantic_child_results_with_scores]


        # 2. BM25 search (child dokümanlar üzerinde)
        bm25_child_results_with_scores = []
        if self.bm25 is not None:
            query_tokens = word_tokenize(processed_query.lower())
            query_tokens = [t for t in query_tokens if t.isalnum() and t not in self.stop_words]
            
            if query_tokens:
                bm25_scores = self.bm25.get_scores(query_tokens)
                # BM25 sonuçlarını (document, score) formatına dönüştür
                # Top k kadar olanları alalım ve bir ranking verelim
                ranked_bm25_docs = sorted(zip(self.bm25_documents, bm25_scores), key=lambda x: x[1], reverse=True)[:config.bm25_top_k * 3]
                bm25_child_results_with_scores = [(doc, score) for doc, score in ranked_bm25_docs]

        # 3. RRF ile birleştirme
        # reciprocal_rank_fusion'a list of lists of (Document, score) vermemiz gerekiyor
        all_ranked_lists = [semantic_results_for_rrf, bm25_child_results_with_scores]
        fused_results = reciprocal_rank_fusion(all_ranked_lists, k=config.top_k_retrieval)
        
        # Sadece Document objelerini al
        fused_documents = [doc for doc, score in fused_results]

        if not fused_documents:
            logger.warning("RRF sonrası belge bulunamadı. Boş liste döndürülüyor.")
            return []

        # 4. Reranking (Cross-encoder ile)
        # Reranker için belge içeriği ve sorgu çiftlerini hazırla
        sentence_pairs = [[query, doc.page_content] for doc in fused_documents]
        
        if not sentence_pairs:
            logger.warning("Reranking için sentence_pairs boş. Reranking atlanıyor.")
            return fused_documents[:config.top_k_rerank] # Sadece RRF sonuçlarının başını döndür

        # Rerank skorlarını hesapla
        # Reranker batch boyutunu yönetebilir, ancak çok büyük değilse tek seferde gönderilebilir.
        rerank_scores = self.reranker.predict(sentence_pairs).tolist()

        # Dokümanları rerank skorlarına göre sırala
        reranked_documents_with_scores = sorted(zip(fused_documents, rerank_scores), key=lambda x: x[1], reverse=True)

        # Sadece en iyi N reranked dokümanı al
        final_retrieved_documents = [doc for doc, score in reranked_documents_with_scores[:config.top_k_rerank]]
        
        return final_retrieved_documents


class RAGSystem:
    def __init__(self):
        self.vectorstore: Optional[Chroma] = None
        self.bm25_retriever: Optional[BM25Okapi] = None # HybridRetriever içinde olacak
        self.embeddings = AdvancedEmbeddings(config.embeddings_model_name)
        self.cross_encoder = CrossEncoder(config.cross_encoder_model_name, device=device) # Init sırasında cihaz belirt
        self.llm = OllamaLLM(model=config.llm_model, base_url=config.ollama_base_url, temperature=config.llm_temperature)
        self.conversation_chain = None
        self.chat_history_memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=config.max_memory_length, input_key="question")
        self.documents_for_bm25: List[Document] = [] # BM25 için tüm child belgeleri saklayacak (artık doğrudan Document objeleri)
        self.id_to_document_map: Dict[str, Document] = {} # Parent belgenin hashinden tam belgeye ulaşım için

        self.vectorstore_path = os.path.join(config.vector_store_dir, "chroma_db")
        self.bm25_index_path = os.path.join(config.cache_dir, "bm25_index.pkl") # cache klasörüne alındı
        self.id_map_path = os.path.join(config.cache_dir, "id_map.pkl") # cache klasörüne alındı
        self.parent_store_path = os.path.join(config.cache_dir, "parent_store.pkl") # cache klasörüne alındı
        self.processed_files_meta_path = config.processed_files_meta_file
        self.processed_files_meta = self._load_processed_files_meta()
        self.parent_document_store = InMemoryStore() # ParentDocumentRetriever için store

        self.parent_retriever: Optional[ParentDocumentRetriever] = None
        self.hybrid_retriever: Optional[HybridRetriever] = None

        logger.info("RAGSystem başlatılıyor...")
        self.load_knowledge_base()
        self.initialize_llm_chain()
        logger.info("RAGSystem başarıyla başlatıldı.")

    def _load_processed_files_meta(self) -> Dict[str, Dict]:
        """İşlenmiş dosyaların meta verilerini yükler."""
        if os.path.exists(self.processed_files_meta_path):
            try:
                with open(self.processed_files_meta_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"İşlenmiş dosyalar meta verisi yüklenemedi: {e}. Boş meta veri ile devam ediliyor.")
                return {}
        return {}

    def _save_processed_files_meta(self):
        """İşlenmiş dosyaların meta verilerini kaydeder."""
        os.makedirs(os.path.dirname(self.processed_files_meta_path), exist_ok=True)
        with open(self.processed_files_meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.processed_files_meta, f, indent=4)

    def _get_file_hash(self, file_path: str) -> str:
        """Dosyanın MD5 hash'ini hesaplar."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        except Exception as e:
            logger.error(f"Dosya hash'i hesaplanırken hata oluştu {file_path}: {e}")
            return "" # Hata durumunda boş döndür
        return hash_md5.hexdigest()

    def _get_text_splitter(self, chunk_size: int, chunk_overlap: int):
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
    
    def _initialize_parent_document_retriever(self):
        """ParentDocumentRetriever'ı başlatır veya yeniden başlatır."""
        # Chroma'nın başlatılmış olması gerekiyor
        if self.vectorstore is None:
            logger.error("ParentDocumentRetriever başlatılamadı: Chroma vectorstore mevcut değil.")
            return

        child_splitter = self._get_text_splitter(config.child_chunk_size, config.child_chunk_overlap)
        parent_splitter = self._get_text_splitter(config.parent_chunk_size, config.parent_chunk_overlap)

        self.parent_retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.parent_document_store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            search_kwargs={'k': config.vector_top_k} # ParentDocumentRetriever'ın kendi arama k'sı
        )
        logger.info("ParentDocumentRetriever başlatıldı.")

    def add_documents_to_knowledge_base(self, documents: List[Document], force_rebuild: bool = False):
        if not documents:
            logger.warning("Eklenecek belge bulunamadı.")
            return

        logger.info(f"{len(documents)} adet belge işleniyor...")

        if force_rebuild:
            logger.info("Zorla yeniden oluşturma: Mevcut bilgi bankası temizleniyor.")
            if os.path.exists(self.vectorstore_path):
                shutil.rmtree(self.vectorstore_path)
            if os.path.exists(self.bm25_index_path):
                os.remove(self.bm25_index_path)
            if os.path.exists(self.id_map_path):
                os.remove(self.id_map_path)
            if os.path.exists(self.parent_store_path):
                os.remove(self.parent_store_path)
            if os.path.exists(self.processed_files_meta_path):
                os.remove(self.processed_files_meta_path)
            
            self.vectorstore = None
            self.documents_for_bm25 = []
            self.id_to_document_map = {}
            self.processed_files_meta = {}
            self.parent_document_store = InMemoryStore() # Sıfırla
            logger.info("Mevcut Chroma veritabanı, BM25 indeksi ve meta verileri temizlendi.")
        
        # Chroma'yı (yeniden) başlat / Eğer yoksa oluştur
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                persist_directory=self.vectorstore_path,
                embedding_function=self.embeddings
            )
            logger.info("Chroma veritabanı başlatıldı.")
        
        # ParentDocumentRetriever'ı başlat
        if self.parent_retriever is None or force_rebuild: # Her zaman yeniden başlat veya ilk kez başlat
            self._initialize_parent_document_retriever()
        
        new_docs_to_add = []
        for doc in documents:
            file_path = doc.metadata.get("source") # Langchain loader'lar "source" metadata'sı ekler
            if file_path:
                file_hash = self._get_file_hash(file_path)
                last_modified = os.path.getmtime(file_path) # Son değişiklik zamanı

                if file_path in self.processed_files_meta and \
                   self.processed_files_meta[file_path]["hash"] == file_hash and \
                   self.processed_files_meta[file_path]["last_modified"] >= last_modified:
                    logger.info(f"'{os.path.basename(file_path)}' zaten işlenmiş ve değişmemiş. Atlanıyor.")
                else:
                    new_docs_to_add.append(doc)
                    self.processed_files_meta[file_path] = {
                        "hash": file_hash,
                        "last_modified": last_modified,
                        "processed_at": datetime.now().isoformat()
                    }
            else:
                # Kaynak bilgisi olmayan belgeleri de ekle
                new_docs_to_add.append(doc)
                logger.warning(f"Belge için kaynak yolu bulunamadı: {doc.metadata.get('id', 'Bilinmeyen belge')}. Değişiklikler takip edilemeyecek.")


        if not new_docs_to_add:
            logger.info("Hiçbir yeni veya güncellenmiş belge bulunamadı. Bilgi bankası güncellenmedi.")
            return

        logger.info(f"{len(new_docs_to_add)} adet yeni/güncellenmiş belge bilgi bankasına ekleniyor...")
        
        # ParentDocumentRetriever'a belgeleri ekle
        self.parent_retriever.add_documents(new_docs_to_add)
        
        # Chroma'yı kalıcı hale getir
        self.vectorstore.persist()
        logger.info("Chroma veritabanı kalıcı hale getirildi.")

        # BM25 için child belgeleri güncelle
        # ParentDocumentRetriever'ın child belgelerini almanın doğrudan bir yolu yok,
        # bu yüzden child splitter ile belgeleri yeniden parçalamamız gerekiyor.
        # Veya ParentDocumentRetriever'ın docstore'undan child ID'lerini alıp,
        # bu ID'lerle child belgelerin content'ini toplayabiliriz.
        # Basitlik için, şimdilik tüm belgeleri tekrar child olarak parçalayalım.
        
        # Mevcut tüm parent dokümanları docstore'dan al
        all_parent_ids = list(self.parent_document_store.yield_keys())
        all_parent_docs = self.parent_document_store.mget(all_parent_ids)
        
        self.documents_for_bm25 = [] # BM25 için listeyi sıfırla
        self.id_to_document_map = {} # Haritayı sıfırla

        child_splitter = self._get_text_splitter(config.child_chunk_size, config.child_chunk_overlap)

        for parent_doc in all_parent_docs:
            if parent_doc: # None olmadığından emin ol
                # Parent belgenin MD5 hash'ini ID olarak kullan
                parent_doc_id = hashlib.md5(parent_doc.page_content.encode('utf-8')).hexdigest()
                self.id_to_document_map[parent_doc_id] = parent_doc # Parent belgeyi haritaya ekle

                child_chunks = child_splitter.split_documents([parent_doc])
                for chunk in child_chunks:
                    # Child'a parent ID'si ekle
                    chunk.metadata["parent_doc_id"] = parent_doc_id
                    self.documents_for_bm25.append(chunk) # BM25 için Document objesini ekle
        
        # BM25 indeksini güncelle
        if self.documents_for_bm25:
            # BM25 için sadece content'leri tokenize et
            tokenized_corpus = [word_tokenize(doc.page_content.lower()) for doc in self.documents_for_bm25]
            self.bm25_retriever = BM25Okapi(tokenized_corpus)
            logger.info(f"BM25 indeksi {len(self.documents_for_bm25)} child belge ile güncellendi.")
        else:
            self.bm25_retriever = None
            logger.warning("BM25 indeksi için belge bulunamadı.")
        
        # HybridRetriever'ı yeniden başlat
        self.hybrid_retriever = HybridRetriever(self.parent_retriever, self.documents_for_bm25, self.embeddings)
        logger.info("HybridRetriever yeniden başlatıldı.")

        self.save_knowledge_base_state()
        self._save_processed_files_meta() # İşlenmiş dosyalar meta verisini kaydet
        logger.info(f"Bilgi bankasına {len(new_docs_to_add)} yeni/güncellenmiş belge eklendi.")

    def load_knowledge_base(self):
        """Kayıtlı bilgi bankasını yükler."""
        # Chroma yükle
        if os.path.exists(self.vectorstore_path) and len(os.listdir(self.vectorstore_path)) > 0:
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.vectorstore_path,
                    embedding_function=self.embeddings
                )
                logger.info("Chroma veritabanı yüklendi.")
            except Exception as e:
                logger.error(f"Chroma veritabanı yüklenirken hata oluştu: {e}. Yeniden oluşturulması gerekebilir.")
                self.vectorstore = None
        else:
            logger.warning("Chroma veritabanı bulunamadı veya boş. Yeni bir veritabanı oluşturulacak.")
            self.vectorstore = None # None olarak kalsın, add_documents_to_knowledge_base oluşturacak

        # BM25 indeksi ve ID haritasını yükle
        if os.path.exists(self.bm25_index_path) and os.path.exists(self.id_map_path) and os.path.exists(self.parent_store_path):
            try:
                with open(self.bm25_index_path, 'rb') as f:
                    self.bm25_retriever = pickle.load(f)
                with open(self.id_map_path, 'rb') as f:
                    self.id_to_document_map = pickle.load(f)
                with open(self.parent_store_path, 'rb') as f:
                    self.parent_document_store = pickle.load(f)
                
                # documents_for_bm25 listesini yeniden oluştur
                self.documents_for_bm25 = list(self.id_to_document_map.values()) # Veya daha doğru bir şekilde child dokümanları yükle

                logger.info("BM25 indeksi, ID haritası ve Parent Belge Deposu yüklendi.")
            except Exception as e:
                logger.error(f"BM25 indeksi, ID haritası veya Parent Belge Deposu yüklenirken hata oluştu: {e}. Yeniden oluşturulması gerekebilir.")
                self.bm25_retriever = None
                self.id_to_document_map = {}
                self.parent_document_store = InMemoryStore() # Hata durumunda sıfırla
                self.documents_for_bm25 = [] # Sıfırla
        else:
            logger.warning("BM25 indeksi, ID haritası veya Parent Belge Deposu bulunamadı. Yeni oluşturulacak.")
            self.bm25_retriever = None
            self.id_to_document_map = {}
            self.parent_document_store = InMemoryStore() # Yeni oluştur
            self.documents_for_bm25 = [] # Yeni oluştur

        # ParentDocumentRetriever ve HybridRetriever'ı yükleme sonrası başlat
        self._initialize_parent_document_retriever()
        # BM25 retriever'ı HybridRetriever'a geçirmeden önce bm25_documents'ı dolduralım.
        # Burada ParentDocumentRetriever'ın docstore'undan ana belgeleri alıp,
        # bunları child splitter ile yeniden parçalayarak bm25_documents'ı doldurmalıyız.
        all_parent_ids = list(self.parent_document_store.yield_keys())
        all_parent_docs = self.parent_document_store.mget(all_parent_ids)
        
        self.documents_for_bm25 = []
        child_splitter = self._get_text_splitter(config.child_chunk_size, config.child_chunk_overlap)
        for parent_doc in all_parent_docs:
            if parent_doc:
                child_chunks = child_splitter.split_documents([parent_doc])
                for chunk in child_chunks:
                    # Child'a parent ID'si ekle (yükleme sırasında tekrar ekleyelim)
                    parent_doc_id = hashlib.md5(parent_doc.page_content.encode('utf-8')).hexdigest()
                    chunk.metadata["parent_doc_id"] = parent_doc_id
                    self.documents_for_bm25.append(chunk)

        # BM25 retriever'ı yükleme sonrası yeniden oluştur (HybridRetriever içinde)
        self.hybrid_retriever = HybridRetriever(self.parent_retriever, self.documents_for_bm25, self.embeddings)
        logger.info("HybridRetriever yüklendi/başlatıldı.")


    def save_knowledge_base_state(self):
        """BM25 indeksi, ID haritası ve Parent Belge Deposu durumunu kaydeder."""
        os.makedirs(config.cache_dir, exist_ok=True) # Cache dizini mevcut olsun

        # BM25 indeksini kaydet
        if self.bm25_retriever:
            with open(self.bm25_index_path, 'wb') as f:
                pickle.dump(self.bm25_retriever, f)
            logger.info("BM25 indeksi kaydedildi.")
        else:
            logger.warning("BM25 indeksi boş, kaydedilemedi.")
            if os.path.exists(self.bm25_index_path): # Eğer boşsa eski dosyayı sil
                os.remove(self.bm25_index_path)

        # ID haritasını kaydet
        with open(self.id_map_path, 'wb') as f:
            pickle.dump(self.id_to_document_map, f)
        logger.info("ID haritası kaydedildi.")

        # Parent Document Store'u kaydet
        if self.parent_document_store:
            with open(self.parent_store_path, 'wb') as f:
                pickle.dump(self.parent_document_store, f)
            logger.info("Parent Document Store kaydedildi.")
        else:
            logger.warning("Parent Document Store boş, kaydedilemedi.")
            if os.path.exists(self.parent_store_path):
                os.remove(self.parent_store_path)


    def initialize_llm_chain(self):
        """LLM zincirini başlatır."""
        self.llm = OllamaLLM(model=config.llm_model, base_url=config.ollama_base_url, temperature=config.llm_temperature)
        rag_prompt = PromptTemplate(
            template=config.rag_template,
            input_variables=["chat_history", "context", "question"]
        )
        self.conversation_chain = LLMChain(
            llm=self.llm,
            prompt=rag_prompt,
            verbose=False, # Daha az çıktı için False
            memory=self.chat_history_memory,
            callbacks=[],
        )
        logger.info("LLM zinciri yeniden başlatıldı.")

    def retrieve_documents(self, query: str) -> List[Document]:
        """Sorguya göre ilgili belgeleri getirir (HybridRetriever kullanılarak)."""
        if self.hybrid_retriever is None:
            logger.error("HybridRetriever başlatılmamış. Bilgi bankası yeniden oluşturulmalı veya yüklenmeli.")
            return []
        
        retrieved_docs = self.hybrid_retriever.retrieve(query)
        logger.info(f"Hybrid Retriever tarafından {len(retrieved_docs)} belge getirildi.")
        return retrieved_docs

    def generate_response(self, query: str, context: str) -> str:
        """LLM kullanarak yanıtı oluşturur."""
        try:
            # LLM zincirinin input key'i "question" olarak ayarlandığından,
            # query ve context'i uygun şekilde iletmeliyiz.
            # Konuşma geçmişi zaten memory tarafından yönetiliyor.
            response = self.conversation_chain.invoke({
                "context": context,
                "question": query,
                "chat_history": self.chat_history_memory.buffer # chat_history doğrudan prompt'a geçiriliyor
            })
            return response['text']
        except Exception as e:
            logger.error(f"Yanıt oluşturulurken hata oluştu: {e}")
            return "Üzgünüm, bir sorun oluştu. Lütfen daha sonra tekrar deneyin."

    def generate_suggestions(self, chat_history: str, answer: str) -> List[str]:
        """Takip soruları önerileri oluşturur."""
        try:
            suggestion_llm = OllamaLLM(model=config.suggestion_llm_model, base_url=config.ollama_base_url, temperature=config.suggestion_temperature)
            suggestion_prompt = PromptTemplate(
                template=config.suggestion_template,
                input_variables=["chat_history", "answer"]
            )
            suggestion_chain = LLMChain(llm=suggestion_llm, prompt=suggestion_prompt)
            
            response = suggestion_chain.invoke({
                "chat_history": chat_history,
                "answer": answer
            })
            suggestions_text = response['text'].strip()
            if suggestions_text.lower() == "boş liste döndür": # LLM'in bu çıktıyı vermesi durumunda
                return []
            return [s.strip() for s in suggestions_text.split('\n') if s.strip()]
        except Exception as e:
            logger.error(f"Takip soruları oluşturulurken hata oluştu: {e}")
            return []


# RAG Sistemi örneği
rag_system = RAGSystem()

def get_chat_history_html():
    """Sohbet geçmişini HTML formatında döndürür."""
    history = rag_system.chat_history_memory.buffer
    html_history = ""
    for i, message in enumerate(history):
        if isinstance(message, tuple): # Langchain 0.1.x ve önceki sürümlerde böyle olabilir
            speaker = "Kullanıcı" if i % 2 == 0 else "Asistan"
            content = message[1]
        else: # Langchain 0.2.x ve sonrası için BaseMessage objeleri
            speaker = "Kullanıcı" if message.type == "human" else "Asistan"
            content = message.content
        
        if speaker == "Kullanıcı":
            html_history += f"<p style='color: blue;'><b>Kullanıcı:</b> {content}</p>"
        else:
            html_history += f"<p style='color: green;'><b>Asistan:</b> {content}</p>"
    return html_history


def respond(message: str, chat_history: List[List[str]]) -> Tuple[str, List[List[str]], List[str]]:
    """Gradio arayüzünden gelen mesaja yanıt verir."""
    logger.info(f"Gelen mesaj: {message}")

    # Sohbet geçmişini güncel (Langchain memory'si zaten yönetiyor)
    # Gradio'nun chat_history'si sadece UI için tutuluyor.
    # rag_system.chat_history_memory.save_context({"question": message}, {"answer": "..."})

    # Belge araması yap
    retrieved_docs = rag_system.retrieve_documents(message)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    if not context:
        logger.warning("İlgili belge bulunamadı.")
        rag_system.chat_history_memory.save_context({"question": message}, {"answer": "Üzgünüm, bu konu hakkında yeterli bilgiye sahip değilim. Lütfen başka bir soru sorun veya bilgi bankasını güncelleyin."})
        full_response = "Üzgünüm, bu konu hakkında yeterli bilgiye sahip değilim. Lütfen başka bir soru sorun veya bilgi bankasını güncelleyin."
        suggestions = rag_system.generate_suggestions(get_chat_history_html(), full_response)
        return full_response, rag_system.chat_history_memory.buffer, suggestions # Gradio formatına uygun döndür

    logger.info(f"LLM'e gönderilen bağlam uzunluğu: {len(context)} karakter.")
    
    # LLM ile yanıt oluştur
    full_response = rag_system.generate_response(message, context)
    
    # Sohbet geçmişini kaydet (Langchain ConversationBufferWindowMemory zaten yapıyor)
    # Gradio'nun chat_history'sini güncelleyelim.
    # rag_system.chat_history_memory.save_context({"question": message}, {"answer": full_response}) # Zaten LLMChain tarafından yapılıyor
    
    # Takip soruları oluştur
    suggestions = rag_system.generate_suggestions(get_chat_history_html(), full_response)

    # Gradio formatında döndür
    # Gradio'nun chat_history formatına uymak için
    # rag_system.chat_history_memory.buffer'daki mesajları List[List[str]] formatına dönüştürmeliyiz.
    gradio_chat_history = []
    for i in range(0, len(rag_system.chat_history_memory.buffer), 2):
        if i+1 < len(rag_system.chat_history_memory.buffer):
            user_msg = rag_system.chat_history_memory.buffer[i].content
            ai_msg = rag_system.chat_history_memory.buffer[i+1].content
            gradio_chat_history.append([user_msg, ai_msg])

    return full_response, gradio_chat_history, suggestions


def rebuild_knowledge_base_full(progress=gr.Progress()):
    """Bilgi bankasını sıfırdan yeniden oluşturur."""
    progress(0, desc="Bilgi bankası temizleniyor...")
    logger.info("Bilgi bankası tam yeniden oluşturma başlatıldı.")
    try:
        # data_dir yerine documents_dir kullan
        loader = DirectoryLoader(
            config.documents_dir,
            glob="**/*",
            loader_cls=lambda path: PyPDFLoader(path) if path.endswith('.pdf') else TextLoader(path, encoding='utf-8')
        )
        documents = loader.load()
        logger.info(f"{len(documents)} adet belge yüklendi.")

        if not documents:
            logger.warning("Yeniden oluşturulacak belge bulunamadı. Lütfen 'documents' klasörüne belge ekleyin.")
            return "<p style='color: orange;'>⚠️ Yeniden oluşturulacak belge bulunamadı. Lütfen 'documents' klasörüne belge ekleyin.</p>"

        progress(0.3, desc="Belgeler işleniyor ve indeksleniyor...")
        rag_system.add_documents_to_knowledge_base(documents, force_rebuild=True)
        rag_system.initialize_llm_chain() # LLM'i bilgi bankası güncellendikten sonra yeniden başlat
        progress(1, desc="Tamamlama")
        return f"<p style='color: green;'>✅ Bilgi bankası başarıyla yeniden oluşturuldu! ({len(rag_system.documents_for_bm25)} güncel child belge parçası)</p>"
    except Exception as e:
        logger.error(f"❌ Tam yeniden oluşturma başarısız oldu: {e}", exc_info=True)
        return f"<p style='color: red;'>❌ Hata: {str(e)}</p>"

def update_knowledge_base_incremental(progress=gr.Progress()):
    """Bilgi bankasını artımlı olarak günceller."""
    progress(0, desc="Yeni veya güncellenmiş belgeler aranıyor...")
    logger.info("Bilgi bankası artımlı güncelleme başlatıldı.")
    try:
        current_files = []
        for root, _, files in os.walk(config.documents_dir): # documents_dir kullan
            for file in files:
                file_path = os.path.join(root, file)
                # UTF-8 BOM'dan temizle
                if file.endswith(('.txt', '.md')): # Sadece metin dosyalarını temizle
                    try:
                        convert_to_utf8_no_bom(file_path)
                    except Exception as e:
                        logger.warning(f"UTF-8 BOM temizlenirken hata oluştu {file_path}: {e}")
                current_files.append(file_path)

        new_or_modified_documents = []
        for file_path in current_files:
            file_hash = rag_system._get_file_hash(file_path)
            last_modified = os.path.getmtime(file_path)

            if file_path not in rag_system.processed_files_meta or \
               rag_system.processed_files_meta[file_path]["hash"] != file_hash or \
               rag_system.processed_files_meta[file_path]["last_modified"] < last_modified:
                
                logger.info(f"Yeni veya güncellenmiş belge tespit edildi: {os.path.basename(file_path)}")
                if file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                elif file_path.endswith(('.txt', '.md')):
                    loader = TextLoader(file_path, encoding='utf-8')
                else:
                    logger.warning(f"Desteklenmeyen dosya türü atlanıyor: {file_path}")
                    continue
                
                try:
                    loaded_docs = loader.load()
                    for doc in loaded_docs:
                        doc.metadata["source"] = file_path # Kaynak yolunu metadata'ya ekle
                    new_or_modified_documents.extend(loaded_docs)
                except Exception as e:
                    logger.error(f"Dosya yüklenirken hata oluştu {file_path}: {e}")
                    continue
        
        if not new_or_modified_documents:
            logger.info("Hiçbir yeni veya güncellenmiş belge bulunamadı. Artımlı güncelleme gerekli değil.")
            return "<p style='color: blue;'>ℹ️ Hiçbir yeni veya güncellenmiş belge bulunamadı. Bilgi bankası güncel.</p>"

        progress(0.3, desc="Yeni belgeler işleniyor ve indeksleniyor...")
        rag_system.add_documents_to_knowledge_base(new_or_modified_documents, force_rebuild=False)
        rag_system.initialize_llm_chain()
        progress(1, desc="Tamamlama")
        return f"<p style='color: green;'>✅ Bilgi bankası artımlı olarak güncellendi! ({len(rag_system.documents_for_bm25)} güncel child belge parçası)</p>"
    except Exception as e:
        logger.error(f"❌ Artımlı güncelleme başarısız oldu: {e}", exc_info=True)
        return f"<p style='color: red;'>❌ Hata: {str(e)}</p>"

def clear_chat():
    """Sohbet geçmişini ve LLM belleğini temizler."""
    rag_system.chat_history_memory.clear()
    logger.info("Sohbet geçmişi temizlendi.")
    return "", [] # Gradio sohbet geçmişini ve metin kutusunu temizle

def create_gradio_interface():
    """Gradio arayüzünü oluşturur."""
    with gr.Blocks(title="Gelişmiş RAG Sistemi") as demo:
        gr.Markdown("# Gelişmiş RAG Sistemi 🤖")
        gr.Markdown(
            "Bu sistem, özel belgeleriniz üzerinde sohbet etmenizi sağlayan bir RAG (Retrieval Augmented Generation) uygulamasıdır. "
            "Arama yapmak ve yanıt üretmek için melez alma (BM25 + Vektör Araması), reranking ve RRF kullanır."
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=500,
                    label="Sohbet Geçmişi",
                    avatar_images=(None, "https://raw.githubusercontent.com/gradio-app/gradio/main/assets/logo.png")
                )
                msg = gr.Textbox(
                    label="Mesajınız",
                    placeholder="Buraya yazın...",
                    container=True,
                    scale=7
                )
                
                with gr.Row():
                    send_btn = gr.Button("Gönder", variant="primary", scale=1)
                    clear_btn = gr.Button("Sohbeti Temizle", scale=1)
                
                suggestions_output = gr.Dataset(
                    label="Takip Soruları Önerileri",
                    components=[gr.Textbox(visible=False)],
                    samples=[],
                    type="array"
                )

            with gr.Column(scale=1):
                with gr.Accordion("Bilgi Bankası Yönetimi", open=True):
                    gr.Markdown("#### Belgelerinizi 'documents' klasörüne yükleyin.")
                    kb_status = gr.Markdown("Bilgi bankası durumu: Yüklendi.")
                    rebuild_btn = gr.Button("Bilgi Bankasını Yeniden Oluştur (Tam)")
                    update_btn = gr.Button("Bilgi Bankasını Güncelle (Artımlı)")
                
                with gr.Accordion("Ayarlar", open=False):
                    llm_model_name = gr.Textbox(
                        label="LLM Modeli (Ollama)",
                        value=config.llm_model,
                        interactive=True
                    )
                    ollama_base_url_input = gr.Textbox(
                        label="Ollama Base URL",
                        value=config.ollama_base_url,
                        interactive=True
                    )
                    embeddings_model_name_input = gr.Textbox(
                        label="Embedding Modeli",
                        value=config.embeddings_model_name,
                        interactive=True
                    )
                    cross_encoder_model_name_input = gr.Textbox(
                        label="Cross-Encoder Modeli",
                        value=config.cross_encoder_model_name,
                        interactive=True
                    )
                    parent_chunk_size_slider = gr.Slider(
                        minimum=500, maximum=4000, value=config.parent_chunk_size,
                        label="Parent Chunk Boyutu", step=100
                    )
                    parent_chunk_overlap_slider = gr.Slider(
                        minimum=0, maximum=1000, value=config.parent_chunk_overlap,
                        label="Parent Chunk Çakışması", step=50
                    )
                    child_chunk_size_slider = gr.Slider(
                        minimum=50, maximum=1000, value=config.child_chunk_size,
                        label="Child Chunk Boyutu", step=10
                    )
                    child_chunk_overlap_slider = gr.Slider(
                        minimum=0, maximum=200, value=config.child_chunk_overlap,
                        label="Child Chunk Çakışması", step=5
                    )
                    bm25_top_k_slider = gr.Slider(
                        minimum=1, maximum=20, value=config.bm25_top_k,
                        label="BM25 Top K", step=1
                    )
                    vector_top_k_slider = gr.Slider(
                        minimum=1, maximum=20, value=config.vector_top_k,
                        label="Vector Top K", step=1
                    )
                    rerank_top_n_slider = gr.Slider(
                        minimum=1, maximum=10, value=config.top_k_rerank,
                        label="Rerank Top N", step=1
                    )
                    llm_temperature_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=config.llm_temperature,
                        label="LLM Sıcaklığı", step=0.1
                    )
                    max_memory_length_slider = gr.Slider(
                        minimum=1, maximum=20, value=config.max_memory_length,
                        label="Sohbet Geçmişi Uzunluğu", step=1
                    )
                    update_params_btn = gr.Button("Parametreleri Güncelle")


        # Olay Dinleyicileri
        msg.submit(
            fn=respond,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, suggestions_output],
            queue=False
        )
        send_btn.click(
            fn=respond,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, suggestions_output],
            queue=False
        )
        clear_btn.click(
            fn=clear_chat,
            inputs=[],
            outputs=[msg, chatbot, suggestions_output], # suggestions_output'ı da temizle
            queue=False
        )
        suggestions_output.click(
            lambda x: x, # Tıklanan öğenin değerini döndürür
            inputs=[suggestions_output],
            outputs=[msg] # Tıklanan öneriyi mesaj kutusuna koyar
        )

        rebuild_btn.click(
            fn=rebuild_knowledge_base_full,
            outputs=kb_status
        )
        update_btn.click(
            fn=update_knowledge_base_incremental,
            outputs=kb_status
        )

        # Ayarlar güncellendiğinde
        update_params_btn.click(
            fn=lambda llm_m, ollama_url, emb_m, cross_m, pc_s, pc_o, cc_s, cc_o, bm25_tk, vec_tk, rerank_tn, llm_temp, max_mem_len: (
                setattr(config, 'llm_model', llm_m),
                setattr(config, 'ollama_base_url', ollama_url),
                setattr(config, 'embeddings_model_name', emb_m),
                setattr(config, 'cross_encoder_model_name', cross_m),
                setattr(config, 'parent_chunk_size', pc_s),
                setattr(config, 'parent_chunk_overlap', pc_o),
                setattr(config, 'child_chunk_size', cc_s),
                setattr(config, 'child_chunk_overlap', cc_o),
                setattr(config, 'bm25_top_k', bm25_tk),
                setattr(config, 'vector_top_k', vec_tk),
                setattr(config, 'top_k_rerank', rerank_tn), # top_k_rerank olarak güncellendi
                setattr(config, 'llm_temperature', llm_temp),
                setattr(config, 'max_memory_length', max_mem_len), # max_memory_length güncellendi
                rag_system.initialize_llm_chain(), # LLM'i yeni sıcaklıkla ve bellek uzunluğuyla yeniden başlat
                rag_system.__init__(), # RAGSystem'i tamamen yeniden başlat (embedding modelleri için)
                "<p style='color: green;'>✅ Parametreler güncellendi ve sistem yeniden yüklendi!</p>"
            ),
            inputs=[
                llm_model_name,
                ollama_base_url_input,
                embeddings_model_name_input,
                cross_encoder_model_name_input,
                parent_chunk_size_slider,
                parent_chunk_overlap_slider,
                child_chunk_size_slider,
                child_chunk_overlap_slider,
                bm25_top_k_slider,
                vector_top_k_slider,
                rerank_top_n_slider,
                llm_temperature_slider,
                max_memory_length_slider
            ],
            outputs=kb_status,
            queue=False
        )

    return demo

def main():
    """Ana fonksiyon"""
    try:
        logger.info("🚀 Gelişmiş RAG Sistemi başlatılıyor...")

        demo = create_gradio_interface()

        if demo is None:
            logger.error("❌ Gradio arayüzü oluşturulamadı!")
            return

        logger.info("🌐 Gradio sunucusu başlatılıyor...")
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False,
            show_error=True,
            inbrowser=True
        )

    except KeyboardInterrupt:
        logger.info("Sistem kapatılıyor...")
    except Exception as e:
        logger.critical(f"Kritik hata: {e}", exc_info=True)

if __name__ == "__main__":
    main()
