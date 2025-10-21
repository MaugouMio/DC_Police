import dataclasses
import datetime
import imagehash
import io
import logging
import os
from PIL import Image
import sqlite3


@dataclasses.dataclass(frozen=True)
class ImageData:
	image_link: str
	source_url: str
	similarity: float


class ImageDHashDatabase:
	"""
	使用差異雜湊 (DHash) 和 SQLite 實現的圖片相似度搜尋資料庫。
	
	相似度計算基於漢明距離 (Hamming Distance)。
	相似度 % = (1 - 漢明距離 / 雜湊長度) * 100 
	
	預設使用 DHash 64 位 (hash_size=8)，即雜湊長度為 64。
	"""

	def __init__(self, db_name: str, hash_size: int = 8):
		# 確保路徑資料夾存在
		os.makedirs(os.path.dirname(db_name), exist_ok=True)
		# 初始化資料庫連線
		self.conn = sqlite3.connect(db_name)
		self.cursor = self.conn.cursor()
		# DHash 通常使用 N x (N+1) 的尺寸來計算 N*N 個雜湊位元
		self.hash_size = hash_size 
		self._create_table()

	def __del__(self):
		self.conn.close()

	def _create_table(self):
		"""
		建立資料庫表格，用於儲存圖片雜湊值和來源網址。
		"""
		self.cursor.execute('''
			CREATE TABLE IF NOT EXISTS images (
				hash TEXT NOT NULL PRIMARY KEY,
				image_link TEXT,
				source_url TEXT,
				message_time TIMESTAMP
			)
		''')
		self.cursor.execute("CREATE INDEX IF NOT EXISTS message_time ON images (message_time)")
		self.conn.commit()

	def _calculate_dhash(self, image: bytes) -> str | None:
		"""
		計算圖片的差異雜湊 (DHash)。
		
		Args:
			image: 二進位圖片檔案。
		
		Returns:
			圖片的 DHash 值 (十六進位字串)。
		"""
		try:
			img = Image.open(io.BytesIO(image))
		except Exception:
			logging.exception("開啟圖片時發生錯誤")
			return None

		try:
			dhash = imagehash.dhash(img, hash_size=self.hash_size)
		except Exception:
			logging.exception("計算雜湊時發生錯誤")
			return None
		
		return str(dhash)

	def store_image(self, *, image: bytes, image_link: str, source_url: str, message_time: datetime.datetime) -> bool:
		"""
		計算圖片雜湊並儲存到資料庫。

		Args:
			image: 二進位圖片檔案。
			image_link: 圖片連結。
			source_url: 圖片的來源網址 (可選)。
		
		Returns:
			儲存成功返回 True，失敗返回 False。
		"""
		dhash_str = self._calculate_dhash(image)
		
		if dhash_str:
			try:
				# 檢查是否已存在相同的 hash 
				self.cursor.execute("SELECT * FROM images WHERE hash = ?", (dhash_str,))
				if self.cursor.fetchone():
					logging.warning("圖片已存在 (Hash: %s)，跳過儲存。", dhash_str)
					return True

				self.cursor.execute(
					"INSERT INTO images (hash, image_link, source_url, message_time) VALUES (?, ?, ?, ?)",
					(dhash_str, image_link, source_url, message_time.timestamp())
				)

				self.conn.commit()
				logging.info("圖片雜湊 %r (來源: %s) 已成功儲存。", dhash_str, source_url)
				return True
			except Exception as e:
				logging.exception("資料庫儲存錯誤")
				return False
		return False

	def delete_image(self, source_url: str) -> bool:
		"""
		根據來源網址刪除圖片記錄。
		
		Args:
			source_url: 圖片的來源網址。
		
		Returns:
			刪除成功返回 True，失敗返回 False。
		"""
		try:
			self.cursor.execute("DELETE FROM images WHERE source_url = ?", (source_url,))
			self.conn.commit()
			logging.info("已刪除來源網址為 %r 的圖片記錄。", source_url)
			return True
		except Exception:
			logging.exception("資料庫刪除錯誤")
			return False

	def search_similar_images(self, query_image: bytes, image_source: str | None = None, similarity_threshold: float = 0.90, top_k: int = 3) -> list[ImageData]:
		"""
		搜尋相似度高於閾值的圖片，並取出前 K 張。
		
		Args:
			query_image: 查詢的二進位圖片。
			similarity_threshold: 相似度閾值 (0.0 到 1.0)。預設 0.90 (90%)。
			top_k: 最多回傳的相似圖片數量。預設 3。
		
		Returns:
			相似圖片的列表，每個元素為 (相似度百分比, 來源網址)。
		"""
		query_hash_str = self._calculate_dhash(query_image)
		if not query_hash_str:
			return []

		query_hash = imagehash.hex_to_hash(query_hash_str)
		
		# 漢明距離計算：
		# 雜湊長度為 hash_size * hash_size (預設 8*8 = 64)
		hash_length = self.hash_size * self.hash_size 
		
		# 漢明距離閾值 (max_distance) = (1 - 相似度閾值) * 雜湊長度
		max_distance = (1 - similarity_threshold) * hash_length
		# 由於漢明距離是整數，取整
		max_distance = int(max_distance) 

		similar_images: list[ImageData] = []

		# 1. 從資料庫取出所有雜湊值
		self.cursor.execute("SELECT hash, image_link, source_url FROM images")
		db_entries = self.cursor.fetchall()

		# 2. 比較漢明距離
		for db_hash_str, image_link, source_url in db_entries:
			if source_url == image_source:
				continue  # 跳過相同來源的圖片

			# 轉換十六進位字串為 imagehash 物件
			db_hash = imagehash.hex_to_hash(db_hash_str)
			
			# 漢明距離 (Hamming Distance) 計算
			distance = query_hash - db_hash

			if distance <= max_distance:
				# 相似度百分比計算
				similarity_percent = (1 - distance / hash_length) * 100
				similar_images.append(
					ImageData(
						image_link=image_link,
						source_url=source_url,
						similarity=similarity_percent
					)
				)

		# 3. 排序並取出前 K 個
		# 依相似度降序排列
		similar_images.sort(key=lambda x: x.similarity, reverse=True)
		
		return similar_images[:top_k]

	def shrink(self, days: int) -> bool:
		"""
		刪除超過指定天數的圖片記錄。
		
		Args:
			days: 保留的天數，超過此天數的記錄將被刪除。
		
		Returns:
			成功返回 True，失敗返回 False。
		"""
		cutoff_time = datetime.datetime.now() - datetime.timedelta(days=days)
		try:
			self.cursor.execute("DELETE FROM images WHERE message_time < ?", (cutoff_time.timestamp(),))
			self.conn.commit()
			logging.info("已刪除超過 %d 天的圖片記錄。", days)
			return True
		except Exception:
			logging.exception("資料庫縮減時發生錯誤")
			return False
