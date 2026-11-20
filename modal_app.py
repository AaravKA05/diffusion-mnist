import modal

app = modal.App("diffusionlab-api")

# Build a lightweight CPU inference environment.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.15.0",
        "torchvision==0.29.2",
        "fastapi==0.162.5",
        "uvicorn==0.54.0",
        "matplotlib==3.11.2",
        "pillow",
    )
    .add_local_python_source(
        "server2",
        "config",
        "diffusion",
        "evaluate",
        "model",
        "sample_cond",
        "data",
    )
    .add_local_dir("checkpoints", remote_path="/root/checkpoints")
    .add_local_dir("static", remote_path="/root/static")
    .add_local_dir("results", remote_path="/root/results")
)


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=900,
)
@modal.asgi_app()
def fastapi_app():
    import server2

    return server2.app
