import dataclasses
import datetime
import io
import logging
import os
import re
import sqlite3
import threading
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import faiss

try:
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import load_image
except ImportError:
    LightGlue = None
    SuperPoint = None

# 為了向下相容既有 imagehash
try:
    import imagehash
except ImportError:
    imagehash = None


@dataclasses.dataclass(frozen=True)
class ImageData:
    image_link: str
    source_url: str
    similarity: float
    match_count: int = 0
    inlier_ratio: float = 0.0


class SharedVisionModels:
    """全域共享的 AI 視覺模型實例（避免每個伺服器重複載入佔用顯存）"""
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("初始化 AI 模型管理器，使用裝置: %s", self.device)

        # 1. DINOv2 模型
        logging.info("正在載入 DINOv2 (dinov2_vits14)...")
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False).eval().to(self.device)
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 2. LightGlue & SuperPoint 模型
        if SuperPoint is not None and LightGlue is not None:
            logging.info("正在載入 SuperPoint 與 LightGlue...")
            self.extractor = SuperPoint(max_num_keypoints=1024).eval().to(self.device)
            self.matcher = LightGlue(features='superpoint').eval().to(self.device)
        else:
            self.extractor = None
            self.matcher = None
            logging.warning("未安裝 LightGlue 套件，幾何精篩功能將受限。")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_embedding(self, img: Image.Image) -> np.ndarray:
        """提取 384 維 DINOv2 向量並進行 L2 正規化"""
        tensor = self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.dinov2(tensor).cpu().numpy().astype("float32")
        faiss.normalize_L2(emb)
        return emb[0]


class ImageDatabase:
    """
    基於 DINOv2 向量檢索 (Faiss) 與 LightGlue 局部特徵匹配的圖片查重資料庫。
    """
    def __init__(self, db_dir: str):
        self.db_dir = db_dir
        os.makedirs(self.db_dir, exist_ok=True)

        self.db_path = os.path.join(self.db_dir, "image_dinov2.db")
        self.faiss_path = os.path.join(self.db_dir, "dinov2.index")
        self.thumb_dir = os.path.join(self.db_dir, "thumbnails")
        os.makedirs(self.thumb_dir, exist_ok=True)

        self.models = SharedVisionModels.get_instance()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_table()
        self._load_or_create_index()

    def __del__(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def _create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT UNIQUE,
                source_url TEXT,
                image_url TEXT,
                thumbnail_path TEXT,
                message_time TIMESTAMP,
                embedding BLOB
            )
        ''')
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_source_url ON images (source_url)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_message_time ON images (message_time)")
        self.conn.commit()

    def _load_or_create_index(self):
        dim = 384
        if os.path.exists(self.faiss_path):
            try:
                self.index = faiss.read_index(self.faiss_path)
                logging.info("[%s] 已載入 Faiss 索引，收錄 %d 筆向量", self.db_dir, self.index.ntotal)
                return
            except Exception as e:
                logging.warning("[%s] 載入 Faiss 索引失敗，將重新建立: %s", self.db_dir, e)

        base_index = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIDMap(base_index)

        # 從資料庫同步現存向量
        self.cursor.execute("SELECT id, embedding FROM images WHERE embedding IS NOT NULL")
        rows = self.cursor.fetchall()
        if rows:
            embeddings = []
            ids = []
            for row_id, emb_blob in rows:
                emb = np.frombuffer(emb_blob, dtype=np.float32)
                embeddings.append(emb)
                ids.append(row_id)
            embeddings = np.array(embeddings, dtype=np.float32)
            self.index.add_with_ids(embeddings, np.array(ids, dtype=np.int64))
            self._save_index()
            logging.info("[%s] 重新構建 Faiss 索引完成，收錄 %d 筆向量", self.db_dir, self.index.ntotal)

    def _save_index(self):
        try:
            faiss.write_index(self.index, self.faiss_path)
        except Exception as e:
            logging.error("[%s] 儲存 Faiss 索引失敗: %s", self.db_dir, e)

    def store_image(self, *, image: bytes, image_link: str, source_url: str, message_time: datetime.datetime) -> bool:
        """
        儲存新圖片：存縮圖、提取 DINOv2 向量、存入資料庫與 Faiss 索引。
        """
        try:
            img = Image.open(io.BytesIO(image))
        except Exception:
            logging.exception("開啟圖片時發生錯誤")
            return False

        try:
            # 解析 message_id
            message_id = source_url.strip().split("/")[-1]
            if not message_id:
                message_id = str(int(message_time.timestamp() * 1000))

            # 儲存 512px WebP 縮圖
            thumb_path = os.path.join(self.thumb_dir, f"{message_id}.webp")
            thumb_img = img.copy()
            thumb_img.thumbnail((512, 512), Image.Resampling.LANCZOS)
            thumb_img.save(thumb_path, "WEBP", quality=85)

            # 提取 DINOv2 向量
            emb = self.models.get_embedding(img)

            # 寫入資料庫
            self.cursor.execute('''
                INSERT INTO images (message_id, source_url, image_url, thumbnail_path, message_time, embedding)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (message_id, source_url, image_link, thumb_path, message_time.timestamp(), emb.tobytes()))
            row_id = self.cursor.lastrowid
            self.conn.commit()

            # 加入 Faiss 索引
            emb_batch = np.array([emb], dtype=np.float32)
            id_batch = np.array([row_id], dtype=np.int64)
            self.index.add_with_ids(emb_batch, id_batch)
            self._save_index()

            logging.info("圖片 (來源: %s) 已成功儲存，ID=%d", source_url, row_id)
            return True
        except sqlite3.IntegrityError:
            logging.info("圖片來源網址已存在或重複 (source_url: %s)，跳過儲存", source_url)
            return True
        except Exception:
            logging.exception("儲存圖片至資料庫時發生錯誤")
            return False

    def delete_image(self, source_url: str) -> bool:
        """根據來源網址刪除圖片記錄"""
        try:
            self.cursor.execute("SELECT id, thumbnail_path FROM images WHERE source_url = ?", (source_url,))
            rows = self.cursor.fetchall()
            if not rows:
                return True

            for row_id, thumb_path in rows:
                if thumb_path and os.path.exists(thumb_path):
                    try:
                        os.remove(thumb_path)
                    except OSError:
                        pass
                # 從 Faiss 索引中移除
                try:
                    self.index.remove_ids(np.array([row_id], dtype=np.int64))
                except Exception:
                    pass

            self.cursor.execute("DELETE FROM images WHERE source_url = ?", (source_url,))
            self.conn.commit()
            self._save_index()
            logging.info("已刪除來源網址為 %r 的圖片記錄", source_url)
            return True
        except Exception:
            logging.exception("刪除圖片記錄時發生錯誤")
            return False

    def search_similar_images(
        self,
        query_image: bytes,
        image_source: str | None = None,
        similarity_threshold: float = 0.25,
        top_k: int = 3,
        stage1_candidates: int = 5
    ) -> list[ImageData]:
        """
        雙階段搜尋相似圖片：
        Stage 1: DINOv2 + Faiss 粗篩候選 Top K
        Stage 2: LightGlue 幾何特徵點精篩判定包含/相同
        """
        if self.index.ntotal == 0:
            return []

        try:
            pil_img = Image.open(io.BytesIO(query_image))
        except Exception:
            logging.exception("無法讀取查詢圖片")
            return []

        # 1. 提取查詢圖片向量
        try:
            query_emb = self.models.get_embedding(pil_img).reshape(1, -1)
        except Exception:
            logging.exception("提取 DINOv2 向量時發生錯誤")
            return []

        # 2. Stage 1: Faiss 檢索
        k = min(stage1_candidates, self.index.ntotal)
        scores, ids = self.index.search(query_emb, k)

        candidates = []
        for score, row_id in zip(scores[0], ids[0]):
            if row_id == -1:
                continue
            self.cursor.execute("SELECT id, image_url, source_url, thumbnail_path FROM images WHERE id = ?", (int(row_id),))
            row = self.cursor.fetchone()
            if not row:
                continue
            cid, img_url, src_url, thumb_path = row
            if image_source and src_url == image_source:
                continue  # 跳過相同訊息來源
            candidates.append((score, img_url, src_url, thumb_path))

        if not candidates or self.models.extractor is None or self.models.matcher is None:
            # 若無 LightGlue，降級為純 DINOv2 相似度判斷
            similar_images = []
            for score, img_url, src_url, _ in candidates:
                if score >= 0.85:
                    similar_images.append(ImageData(image_link=img_url, source_url=src_url, similarity=float(score * 100)))
            return similar_images[:top_k]

        # 3. Stage 2: LightGlue 精篩
        similar_images: list[ImageData] = []
        try:
            # 準備查詢圖片的張量
            img0_rgb = pil_img.convert("RGB")
            img0_tensor = T.functional.to_tensor(img0_rgb).unsqueeze(0).to(self.models.device)
            feats0 = self.models.extractor.extract(img0_tensor)
            kpts0_count = len(feats0['keypoints'][0])
        except Exception:
            logging.exception("提取查詢圖片特徵點時發生錯誤")
            return []

        for vec_score, img_url, src_url, thumb_path in candidates:
            if not thumb_path or not os.path.exists(thumb_path):
                continue

            try:
                img1_tensor = load_image(thumb_path).to(self.models.device)
                feats1 = self.models.extractor.extract(img1_tensor)
                kpts1_count = len(feats1['keypoints'][0])

                matches01 = self.models.matcher({'image0': feats0, 'image1': feats1})
                matches = matches01['matches'][0]
                match_count = len(matches)

                min_kpts = min(kpts0_count, kpts1_count)
                inlier_ratio = (match_count / min_kpts) if min_kpts > 0 else 0.0

                # 判定門檻：覆蓋率 >= 25% 或吻合特徵點 >= 50
                is_duplicate = (inlier_ratio >= similarity_threshold) or (match_count >= 50)
                if is_duplicate:
                    # 綜合計算相似度百分比
                    sim_pct = max(inlier_ratio * 100, float(vec_score * 100))
                    similar_images.append(
                        ImageData(
                            image_link=img_url or "",
                            source_url=src_url,
                            similarity=sim_pct,
                            match_count=match_count,
                            inlier_ratio=inlier_ratio
                        )
                    )
            except Exception:
                logging.exception("LightGlue 比對圖片時發生錯誤: %s", thumb_path)

        # 依相似度排序取出前 top_k
        similar_images.sort(key=lambda x: x.similarity, reverse=True)
        return similar_images[:top_k]

    def shrink(self, days: int) -> bool:
        """刪除超過指定天數的圖片記錄"""
        cutoff_time = datetime.datetime.now() - datetime.timedelta(days=days)
        try:
            self.cursor.execute("SELECT id, thumbnail_path FROM images WHERE message_time < ?", (cutoff_time.timestamp(),))
            rows = self.cursor.fetchall()
            for row_id, thumb_path in rows:
                if thumb_path and os.path.exists(thumb_path):
                    try:
                        os.remove(thumb_path)
                    except OSError:
                        pass
                try:
                    self.index.remove_ids(np.array([row_id], dtype=np.int64))
                except Exception:
                    pass

            self.cursor.execute("DELETE FROM images WHERE message_time < ?", (cutoff_time.timestamp(),))
            self.conn.commit()
            self._save_index()
            logging.info("[%s] 已成功刪除超過 %d 天的圖片記錄", self.db_dir, days)
            return True
        except Exception:
            logging.exception("縮減資料庫時發生錯誤")
            return False


# 向下相容保留原有別名
class ImageDHashDatabase(ImageDatabase):
    def __init__(self, db_name: str, hash_size: int = 8):
        # 原本 db_name 格式為 "db/<guild_id>/image_dhash.db"
        # 直接使用其目錄初始化新資料庫
        db_dir = os.path.dirname(db_name)
        super().__init__(db_dir)
