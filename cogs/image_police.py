import argparse
from collections.abc import Sequence
import dataclasses
import datetime
import discord
from discord.ext import commands
from discord import app_commands
import functools
import logging
import re
import urllib.request

from db import ImageData, ImageDHashDatabase


_arg_parser = argparse.ArgumentParser()
_arg_parser.add_argument("--temp_msg_channel", type=str, help="The <guild_id>/<channel_id> of the temporary message channel for discord image fetching.", required=False)
_ARGS = _arg_parser.parse_args()

_DISCORD_MEDIA_PREFIX = "https://media.discordapp.net/"
_DISCORD_MSG_URL_TEMPLATE = "https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"


@dataclasses.dataclass(frozen=True)
class MessageReference:
	message: discord.Message
	channel: discord.TextChannel


class ImagePolice(commands.Cog):
	def __init__(self, bot: commands.Bot):
		self.bot = bot
		self.db: dict[int, ImageDHashDatabase] = {}  # 每個伺服器一個資料庫
	
	def _get_db(self, guild_id: int) -> ImageDHashDatabase:
		if guild_id not in self.db:
			self.db[guild_id] = ImageDHashDatabase(db_name=f"db/{guild_id}/image_dhash.db")
		return self.db[guild_id]
	
	async def _get_image_by_attachment(self, attachment: discord.Attachment) -> bytes | None:
		if any(attachment.filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]):
			try:
				image = await attachment.read()
			except Exception:
				logging.exception(f"無法讀取附件圖片 {attachment.url}")
				return None
			return image
		return None
	
	async def _get_temp_msg_channel(self) -> discord.TextChannel | None:
		if not _ARGS.temp_msg_channel:
			return None
		
		try:
			guild_id, channel_id = _ARGS.temp_msg_channel.split("/")
		except ValueError:
			logging.warning("無效的 temp_msg_channel 參數格式：%s", _ARGS.temp_msg_channel)
			return None
		
		guild = self.bot.get_guild(int(guild_id))
		if not guild:
			return None
		
		channel = guild.get_channel(int(channel_id))
		if not isinstance(channel, discord.TextChannel):
			return None
		
		return channel

	async def _get_image_by_link(self, image_link: str) -> bytes | None:
		if image_link.startswith(_DISCORD_MEDIA_PREFIX):  # Discord 伺服器上的圖片，要透過內建手段刷新網址，直接存取會失敗
			image_link = discord.Embed().set_image(url=image_link)
			temp_channel = await self._get_temp_msg_channel()
			if temp_channel:
				try:
					message = await temp_channel.send("解析圖片", embed=image_link, delete_after=0)
				except discord.HTTPException:
					logging.exception("無法在臨時訊息頻道發送訊息以抓取 Discord 圖片 %r", image_link)
				else:
					image_link = message.embeds[0].image.url
			else:
				logging.warning("無法取得臨時訊息頻道以抓取 Discord 圖片 %r", image_link)
		
		request = urllib.request.Request(image_link, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
		try:
			with urllib.request.urlopen(request) as ret:
				return ret.read()
		except Exception:
			logging.exception("無法讀取圖片連結 %r", image_link)
			return None

	def _generate_image_embeds(self, image_datas: Sequence[ImageData]) -> Sequence[discord.Embed]:
		embeds = []
		for data in image_datas:
			embed = discord.Embed(
				title=f"相似度: {data.similarity:.2f}%",
				description=data.source_url,
			)
			embed.set_image(url=data.image_link)
			embeds.append(embed)
		return embeds

	async def _check_image(self, image: bytes, interaction: discord.Interaction):
		db = self._get_db(interaction.guild_id)
		similar_images = db.search_similar_images(image)

		if similar_images:
			response = "⚠️發現部分相似的圖片⚠️"
			embeds = self._generate_image_embeds(similar_images)
			await interaction.followup.send(response, embeds=embeds)
		else:
			await interaction.followup.send("✅沒有發現相似的圖片")

	@app_commands.command(name="check_image_file", description="檢查圖片是否曾經發過")
	@app_commands.describe(image_file="圖片檔案")
	async def check_image_file(self, interaction: discord.Interaction, image_file: discord.Attachment):
		await interaction.response.defer(ephemeral=True)  # 延遲回應，給予更多時間處理
		
		image = await self._get_image_by_attachment(image_file)
		if not image:
			await interaction.followup.send("無法讀取圖片檔案")
			return

		await self._check_image(image, interaction)

	@app_commands.command(name="check_image_link", description="檢查圖片是否曾經發過(連結)")
	@app_commands.describe(image_link="圖片網址")
	async def check_image_link(self, interaction: discord.Interaction, image_link: str):
		await interaction.response.defer(ephemeral=True)  # 延遲回應，給予更多時間處理

		image = await self._get_image_by_link(image_link)
		if not image:
			await interaction.followup.send("無法讀取圖片連結")
			return
		
		await self._check_image(image, interaction)

	async def _get_images_from_message(self, message: discord.Message) -> list[tuple[bytes, str]]:
		images: list[tuple[bytes, str]] = []
		# 先把附件圖片抓下來
		for attachment in message.attachments:
			image = await self._get_image_by_attachment(attachment)
			if image:
				images.append((image, attachment.url))
		
		# 再把訊息中的圖片連結抓下來
		outer_image_urls = re.findall(r"(https?://[^\s]+?\.(?:png|jpg|jpeg|bmp|webp)(?:\?[^\s]*)?)", message.content, re.IGNORECASE)
		for url in outer_image_urls:
			image = await self._get_image_by_link(url)
			if image:
				images.append((image, url))
		
		return images

	async def _save_images_from_message(self, message: discord.Message) -> list[tuple[bytes, str]]:
		logging.info("Saving images from message %d in guild %d", message.id, message.guild.id)
		db = self._get_db(message.guild.id)

		images = await self._get_images_from_message(message)
		for image, url in images:
			db.store_image(image=image, image_link=url, source_url=message.jump_url, message_time=message.created_at)

		return images

	async def _parse_message_url(self, interaction: discord.Interaction, message_url: str) -> MessageReference | None:
		try:
			guild_id, channel_id, message_id = map(int, message_url.split("/")[-3:])
		except ValueError:
			await interaction.followup.send("無效的訊息連結格式")
			return None

		if guild_id != interaction.guild_id:
			await interaction.followup.send("該訊息不存在於當前伺服器")
			return None
		
		channel = self.bot.get_channel(channel_id)
		if not channel:
			await interaction.followup.send("找不到指定的訊息頻道")
			return None
		
		message = await channel.fetch_message(message_id)
		if not message:
			await interaction.followup.send("找不到指定的訊息")
			return None
		
		return MessageReference(message=message, channel=channel)

	@app_commands.command(name="save_images", description="把指定訊息中的圖片記錄下來")
	@app_commands.describe(message_url="訊息連結")
	async def save_images(self, interaction: discord.Interaction, message_url: str):
		await interaction.response.defer(ephemeral=True)  # 延遲回應，給予更多時間處理
		
		message_reference = await self._parse_message_url(interaction, message_url)
		if not message_reference:
			return

		saved_images = await self._save_images_from_message(message_reference.message)
		await interaction.followup.send(f"已儲存 {len(saved_images)} 張圖片")

	async def _sue(self, interaction: discord.Interaction, defendant: discord.Message, evidence: Sequence[discord.Embed]):
		await defendant.reply(f"{interaction.user.mention} 已對你提起重複發圖告訴，以下列舉相關事證：", embeds=evidence)
		await interaction.response.edit_message(content="已成功提告，稍後自動刪除訊息", delete_after=0)

	@app_commands.command(name="censor", description="稽查指定訊息中的圖片")
	@app_commands.describe(message_url="訊息連結")
	async def censor(self, interaction: discord.Interaction, message_url: str):
		await interaction.response.defer(ephemeral=True)  # 延遲回應，給予更多時間處理
		
		message_reference = await self._parse_message_url(interaction, message_url)
		if not message_reference:
			return

		images = await self._get_images_from_message(message_reference.message)
		db = self._get_db(message_reference.message.guild.id)

		# 取出訊息中所有圖片的相關圖片
		similar_images = set()
		for image, _ in images:
			similar_images = similar_images.union(db.search_similar_images(image, image_source=message_reference.message.jump_url))

		if similar_images:
			# 將結果依相似度重新排序
			similar_images = sorted(similar_images, key=lambda x: x.similarity, reverse=True)

			response = "⚠️警方已掌握關鍵事證⚠️"
			embeds = self._generate_image_embeds(similar_images)

			view = discord.ui.View(timeout = 30)
			button = discord.ui.Button(label = "提告")
			button.callback = functools.partial(self._sue, defendant=message_reference.message, evidence=embeds)
			view.add_item(button)

			await interaction.followup.send(response, view=view, embeds=embeds)
		else:
			await interaction.followup.send("缺乏具體事證，無法立案")

	@app_commands.command(name="fetch_channel", description="把頻道中最近的訊息全部記錄下來 (注意！會讓 BOT 非常忙碌！)")
	@app_commands.describe(days="超過此天數的記錄將被刪除")
	@app_commands.default_permissions(moderate_members=True)
	@app_commands.checks.has_permissions(moderate_members=True)
	async def fetch_channel(self, interaction: discord.Interaction, days: int = 180):
		await interaction.response.defer(ephemeral=True)  # 延遲回應，給予更多時間處理
		
		channel = interaction.channel
		if not isinstance(channel, discord.TextChannel):
			await interaction.followup.send("此指令只能在文字頻道中使用")
			return
		
		logging.info("======= Start fetching images from channel %d in guild %d =======", channel.id, interaction.guild_id)

		saved_count = 0
		earliest_date = datetime.datetime.now() - datetime.timedelta(days=days)
		async for message in channel.history(limit=None, after=earliest_date):
			saved_images = await self._save_images_from_message(message)
			saved_count += len(saved_images)
		
		await interaction.followup.send(f"已儲存頻道中 {saved_count} 張圖片")

	@app_commands.command(name="shrink", description="縮減圖片資料庫")
	@app_commands.describe(days="超過此天數的記錄將被刪除")
	@app_commands.default_permissions(moderate_members=True)
	@app_commands.checks.has_permissions(moderate_members=True)
	async def shrink(self, interaction: discord.Interaction, days: int = 180):
		db = self._get_db(interaction.guild_id)
		if db.shrink(days=days):
			await interaction.response.send_message(f"已成功刪除超過 {days} 天的圖片記錄", ephemeral=True)
		else:
			await interaction.response.send_message("縮減圖片資料庫時發生錯誤", ephemeral=True)

	@commands.Cog.listener()
	async def on_message(self, message: discord.Message):
		if message.author.bot:
			return  # 忽略機器人訊息
		
		if not message.guild:
			return  # 忽略私訊
		
		await self._save_images_from_message(message)
	
	@commands.Cog.listener()
	async def on_raw_message_delete(self, payload: discord.RawMessageDeleteEvent):
		if not payload.guild_id:
			return  # 忽略私訊
		
		db = self._get_db(payload.guild_id)
		db.delete_image(
			_DISCORD_MSG_URL_TEMPLATE.format(
				guild_id=payload.guild_id,
				channel_id=payload.channel_id,
				message_id=payload.message_id
			)
		)

	@commands.command()
	async def sync(self, ctx: commands.Context):
		if not ctx.message.author.guild_permissions.moderate_members:
			return
			
		fmt = await self.bot.tree.sync()
		cmd_list = "\n- ".join([cmd.name for cmd in fmt])
		await ctx.send(f"Synced {len(fmt)} commands:\n{cmd_list}")


async def setup(bot: commands.Bot):
	await bot.add_cog(ImagePolice(bot))
