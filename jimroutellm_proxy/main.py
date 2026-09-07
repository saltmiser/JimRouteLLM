import uvicorn
import logging
from jimroutellm_proxy.config import settings

logger = logging.getLogger("jimroutellm.main")

def run():
    print("=" * 65)
    print("  Starting JimRouteLLM Hybrid Proxy Server")
    print("  Router: ModernBERT-Large (395M) | Cloud: Gemini AI Studio")
    print(f"  Listening on: http://{settings.proxy_host}:{settings.proxy_port}")
    print("=" * 65)
    
    uvicorn.run(
        "jimroutellm_proxy.server:app",
        host=settings.proxy_host,
        port=settings.proxy_port,
        reload=False,
        log_level="info",
    )

if __name__ == "__main__":
    run()
