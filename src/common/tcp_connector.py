import ssl
import aiohttp

try:
    import certifi

    _ssl_context = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _ssl_context = ssl.create_default_context()


async def get_tcp_connector() -> aiohttp.TCPConnector:
    return aiohttp.TCPConnector(ssl=_ssl_context)


async def get_ssl_context() -> ssl.SSLContext:
    return _ssl_context
