import base64
from PIL import Image
from io import BytesIO
import os
import importlib.util

def img_to_compressed_base64(image, max_size=(800, 800), quality=70):
    img = Image.open(image)

    img.thumbnail(max_size)

    buffer = BytesIO()
    if img.format == "PNG":
        img.save(buffer, format="PNG", optimize=True)
    else:
        img.save(buffer, format="JPEG", quality=quality)

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded

def fig_to_compressed_base64(fig, resize_height: int = 400) -> str:
    import io, base64
    from PIL import Image
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    img = Image.open(buf)
    w, h = img.size
    new_w = int(w * resize_height / h)

    if hasattr(Image, 'Resampling'):
        resample = Image.Resampling.LANCZOS
    else:
        resample = Image.ANTIALIAS

    img = img.resize((new_w, resize_height), resample)

    buf_resized = io.BytesIO()
    img.save(buf_resized, format='PNG', optimize=True)
    img_str = base64.b64encode(buf_resized.getvalue()).decode()

    plt.close(fig)
    return img_str


def import_function_from_file(abs_path: str, pyfile: str, func_name: str):
    file_path = os.path.join(abs_path, f"{pyfile}.py")

    spec = importlib.util.spec_from_file_location(pyfile, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return getattr(module, func_name)


def import_module_from_file(abs_path: str, pyfile: str):
    file_path = os.path.join(abs_path, f"{pyfile}.py")

    spec = importlib.util.spec_from_file_location(pyfile, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module

def extract_content_from_message(msg):
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Iterate through list items if content is a list
        msg = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    # Process LaTeX-style text
                    if isinstance(item["text"], str):
                        msg.append(item["text"])
                    elif isinstance(item["text"], Exception):
                        msg.append(" ".join(item["text"].args))
            elif isinstance(item, str):
                # Handle plain text items in the list
                msg.append(item)
        return "\n".join(msg)
    elif isinstance(content, dict):
        # Display text if present in a single dictionary
        if "text" in content:
            return content["text"]
        return "No messages"