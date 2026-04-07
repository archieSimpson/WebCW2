"""Web crawler for the COMP3011 search engine."""

import time
import urllib.robotparser
from collections import deque
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


class Crawler:
    """Breadth-first crawler that collects {url: html}."""

    def __init__(self, base_url, politeness_window=6):
        self.base_url = base_url
        self.politeness_window = politeness_window
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

    def crawl(self):
        """Run the BFS crawl and return {url: html}."""
        queue = deque([self.base_url])
        self.visited.add(self.base_url)

        while queue:
            url = queue.popleft()
            if not self._is_allowed(url):
                print(f"Skipping (robots.txt): {url}")
                continue

            try:
                print(f"Crawling: {url}")
                response = self.session.get(url)
                response.raise_for_status()

                if 'text/html' not in response.headers.get('Content-Type', ''):
                    continue

                self.pages[url] = response.text

                soup = BeautifulSoup(response.text, 'html.parser')
                for tag in soup.find_all('a', href=True):
                    link = urljoin(url, tag['href'])
                    link = self._normalise_url(link)
                    if (link not in self.visited
                            and self._is_same_domain(link)
                            and link.startswith('http')):
                        self.visited.add(link)
                        queue.append(link)

                time.sleep(self.politeness_window)

            except requests.RequestException as e:
                print(f"Error crawling {url}: {e}")

        return self.pages
