"""Web crawler for the COMP3011 search engine."""

import urllib.robotparser
from urllib.parse import urljoin, urlparse

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
        self.rp = urllib.robotparser.RobotFileParser()
        self._load_robots_txt()

    def _load_robots_txt(self):
        """Fetch and parse robots.txt; failures are non-fatal."""
        robots_url = urljoin(self.base_url, '/robots.txt')
        try:
            self.rp.set_url(robots_url)
            self.rp.read()
        except Exception:
            pass

    def _is_allowed(self, url):
        """Return True when robots.txt allows fetching url."""
        try:
            return self.rp.can_fetch('*', url)
        except Exception:
            return True

    def _is_same_domain(self, url):
        """Return True when url is on the same netloc as the seed."""
        return urlparse(url).netloc == urlparse(self.base_url).netloc

    def _normalise_url(self, url):
        """Strip fragment and query so equivalent urls collapse to one."""
        return url.split('#')[0].split('?')[0]
