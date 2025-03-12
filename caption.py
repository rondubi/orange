import asyncio
from io import BytesIO
from typing import Literal

import aiohttp
from PIL import Image, ImageDraw, ImageFont
from typing import Optional

ALLOWED_FORMATS = {"png", "jpeg", "jpg"}

def _caption(
    img: Image.Image,
    bottom_text: Optional[str] = None,
    top_text: Optional[str] = None
):
    width, height = img.size
    draw = ImageDraw.ImageDraw(img)
    if top_text is not None:
        font = ImageFont.truetype("Impact.ttf", 60)
        draw.multiline_text(
            (width / 2, height / 10),
            text=top_text,
            stroke_fill=0,
            stroke_width=5.0,
            anchor="mm",
            fill=(255, 255, 255),
            font=font,
        )
    if bottom_text is not None:
        font = ImageFont.truetype("Impact.ttf", 60)
        draw.multiline_text(
            (width / 2, height - height / 10),
            text=bottom_text,
            stroke_fill=0,
            stroke_width=5.0,
            anchor="mm",
            fill=(255, 255, 255),
            font=font,
        )


def _determine_formats_by_content_type(content_type: str) -> list[str]:
    formats = []
    content_type = content_type.casefold()
    if not content_type.startswith("image/"):
        raise ValueError("Template is not an image")
    format = content_type.removeprefix("image/")
    if format not in ALLOWED_FORMATS:
        raise ValueError(f"Template has an unsupported image format: '{format}'")
    formats.append(format)
    return formats


def _process(
    img: Image.Image,
    bottom_text: Optional[str] = None,
    top_text: Optional[str] = None,
    result_format: Literal["png", "jpg"] = "png",
) -> bytes:
    _caption(img, bottom_text=bottom_text, top_text=top_text)

    result = BytesIO()
    img.save(result, format=result_format)

    return result.getbuffer()


async def _fetch_image(url: str) -> Image.Image:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            content_type = response.content_type
            bytes = await response.read()

    def _load(image_bytes, content_type) -> Image.Image:
        formats = _determine_formats_by_content_type(content_type)

        io = BytesIO(image_bytes)
        return Image.open(io, formats=formats)

    return await asyncio.to_thread(_load, bytes, content_type)


async def _get_url(url: str) -> Image.Image:
    if url.startswith("/"):
        return await asyncio.to_thread(Image.open, url)
    elif url.startswith("http"):
        return await _fetch_image(url.removeprefix("http://"))
    raise ValueError("Unsupported URL")


async def caption_template(
    template_url: str,
    bottom_text: Optional[str] = None,
    top_text: Optional[str] = None,
    result_format: Literal["png", "jpg"] = "png",
) -> bytes:
    """
    Add a caption to a template given by the provided url and
    returns the captioned image bytes.
    """
    image = await _get_url(template_url)

    return await asyncio.to_thread(
        _process,
        image,
        bottom_text=bottom_text,
        top_text=top_text,
        result_format=result_format,
    )
