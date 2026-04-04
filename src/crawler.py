"""Web crawler for the COMP3011 search engine."""

from urllib.parse import urlparse

import requests


class Crawler:
    """Breadth-first crawler that collects {url: html}."""

    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update(
            {'User-Agent': 'COMP3011Crawler/1.0'}
        )
        self.pages = {}
        self.visited = set()

    def _is_same_domain(self, url):
        """Return True when url is on the same netloc as the seed."""
        return urlparse(url).netloc == urlparse(self.base_url).netloc
