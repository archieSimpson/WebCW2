"""Web crawler for the COMP3011 search engine."""

import requests


class Crawler:
    """Breadth-first crawler that collects {url: html}."""

    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()
        self.pages = {}
        self.visited = set()
