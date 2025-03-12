import requests
import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import pickle
import logging
from pathlib import Path
import time
import praw
from urllib.parse import urlparse
import random
import json

"""
Ex Initialization:
grabber = MemeGrabber(
    reddit_client_id="YOUR_REDDIT_CLIENT_ID",
    reddit_client_secret="YOUR_REDDIT_CLIENT_SECRET",
    reddit_user_agent="python:meme-grabber:v1.0",
    local_memes_folder="reaction_memes"
)

# Get templates from ImgFlip - semantic search of ~200 classic templates
imgflip_templates = grabber.get_template(
    source="imgflip",
    query="distracted boyfriend",
    limit=2
)

# Get templates from Reddit - semantic search for contemporary templates (may already have captions)
reddit_templates = grabber.get_template(
    source="reddit",
    query="guy thinking with math equations",
    limit=2
)

# Get a reaction meme (from reaction folders)
reaction_meme = grabber.get_template(
    source="reaction",
    query="liar"
)
"""

class MemeGrabberBase:
    """Base class for all meme grabbers with common utilities."""

    def __init__(self, cache_dir: str = "meme_cache"):
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Setup cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)


class RedditMemeGrabber(MemeGrabberBase):
    """Grabber for memes from Reddit."""

    def __init__(self, reddit_client_id: str = None,
                 reddit_client_secret: str = None,
                 reddit_user_agent: str = "python:meme-grabber:v1.0 (by /u/YOUR_USERNAME)",
                 cache_dir: str = "meme_cache",
                 subreddit: str = "MemeTemplatesOfficial",
                 cache_expiry: int = 7):  # Cache expiry in days
        """
        Initialize the RedditMemeGrabber with Reddit API credentials and embedding model.

        Args:
            reddit_client_id: Reddit API client ID
            reddit_client_secret: Reddit API client secret
            reddit_user_agent: Reddit API user agent
            cache_dir: Directory to store cached memes and embeddings
            subreddit: Subreddit to scrape for memes (default: MemeTemplatesOfficial)
            cache_expiry: Number of days after which to refresh the cache
        """
        super().__init__(cache_dir)

        self.subreddit_name = subreddit
        self.cache_expiry = cache_expiry * 86400  # Convert days to seconds

        # Initialize Reddit API client
        self.reddit_client_id = reddit_client_id
        self.reddit_client_secret = reddit_client_secret
        self.reddit_user_agent = reddit_user_agent
        self.reddit = None
        if reddit_client_id and reddit_client_secret:
            self.reddit = praw.Reddit(
                client_id=reddit_client_id,
                client_secret=reddit_client_secret,
                user_agent=reddit_user_agent
            )

        # Initialize embedding model for semantic search
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_available = True
        except ImportError:
            self.logger.warning("sentence-transformers not installed. Falling back to keyword matching.")
            self.embedding_available = False

        # Load cache if it exists
        self.meme_cache = {}
        self.embedding_cache = {}
        self.last_update = 0
        self._load_cache()

    def _load_cache(self):
        """Load meme cache and embeddings from disk if they exist."""
        meme_cache_path = self.cache_dir / f"reddit_{self.subreddit_name}_meme_cache.pkl"
        embedding_cache_path = self.cache_dir / f"reddit_{self.subreddit_name}_embedding_cache.pkl"
        last_update_path = self.cache_dir / f"reddit_{self.subreddit_name}_last_update.txt"

        if meme_cache_path.exists():
            try:
                with open(meme_cache_path, 'rb') as f:
                    self.meme_cache = pickle.load(f)
                self.logger.info(f"Loaded {len(self.meme_cache)} memes from cache")
            except Exception as e:
                self.logger.error(f"Failed to load meme cache: {e}")

        if embedding_cache_path.exists() and self.embedding_available:
            try:
                with open(embedding_cache_path, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                self.logger.info(f"Loaded {len(self.embedding_cache)} embeddings from cache")
            except Exception as e:
                self.logger.error(f"Failed to load embedding cache: {e}")

        if last_update_path.exists():
            try:
                with open(last_update_path, 'r') as f:
                    self.last_update = float(f.read().strip())
            except Exception:
                self.last_update = 0

    def _save_cache(self):
        """Save meme cache and embeddings to disk."""
        meme_cache_path = self.cache_dir / f"reddit_{self.subreddit_name}_meme_cache.pkl"
        embedding_cache_path = self.cache_dir / f"reddit_{self.subreddit_name}_embedding_cache.pkl"
        last_update_path = self.cache_dir / f"reddit_{self.subreddit_name}_last_update.txt"

        try:
            with open(meme_cache_path, 'wb') as f:
                pickle.dump(self.meme_cache, f)

            if self.embedding_available:
                with open(embedding_cache_path, 'wb') as f:
                    pickle.dump(self.embedding_cache, f)

            with open(last_update_path, 'w') as f:
                f.write(str(time.time()))

            self.last_update = time.time()
            self.logger.info("Cache saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")

    def _is_image_url(self, url: str) -> bool:
        """Check if a URL is an image URL."""
        parsed = urlparse(url)
        path = parsed.path.lower()
        return path.endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp'))

    def _extract_image_url(self, submission) -> str:
        """Extract image URL from a Reddit submission."""
        # Direct image posts
        if hasattr(submission, 'url') and self._is_image_url(submission.url):
            return submission.url

        # Reddit-hosted images (preview images)
        if hasattr(submission, 'preview') and 'images' in submission.preview:
            images = submission.preview['images']
            if images and 'source' in images[0]:
                return images[0]['source']['url']

        # Check if the post has any media
        if hasattr(submission, 'media') and submission.media:
            if 'reddit_video' in submission.media:
                return None  # Skip videos

            # Try to extract from media metadata
            if hasattr(submission, 'media_metadata'):
                for media_id in submission.media_metadata:
                    media = submission.media_metadata[media_id]
                    if media['e'] == 'Image':
                        return media['s']['u']

        # Check for gallery posts
        if hasattr(submission, 'is_gallery') and submission.is_gallery:
            if hasattr(submission, 'media_metadata'):
                for media_id in submission.media_metadata:
                    media = submission.media_metadata[media_id]
                    if media['e'] == 'Image':
                        return media['s']['u']

        return None

    def fetch_reddit_memes(self, limit: int = 100, time_filter: str = 'all') -> List[Dict]:
        """
        Fetch memes with "template" flair from Reddit.

        Args:
            limit: Maximum number of submissions to fetch
            time_filter: 'hour', 'day', 'week', 'month', 'year', or 'all'

        Returns:
            List of meme dictionaries with "template" flair
        """
        if not self.reddit:
            self.logger.error("Reddit API credentials not provided")
            return []

        try:
            subreddit = self.reddit.subreddit(self.subreddit_name)
            submissions = subreddit.top(time_filter=time_filter, limit=limit)

            results = []
            for submission in submissions:
                # Skip posts without "template" flair
                if not hasattr(submission, 'link_flair_text') or submission.link_flair_text != "template":
                    continue

                # Skip non-image posts
                image_url = self._extract_image_url(submission)
                if not image_url:
                    continue

                # Basic info
                meme_id = submission.id
                title = submission.title
                score = submission.score  # Upvotes

                # Create meme data dictionary
                meme_data = {
                    'id': meme_id,
                    'url': image_url,
                    'title': title,
                    'description': title,  # Use title as description
                    'upvotes': score,
                    'permalink': f"https://reddit.com{submission.permalink}",
                    'created_utc': submission.created_utc,
                    'source': 'reddit',
                    'flair': submission.link_flair_text
                }

                results.append(meme_data)

                # Add to cache
                self.meme_cache[meme_id] = meme_data

                # Update embedding
                if self.embedding_available:
                    self.update_embedding_cache(meme_id, title)

            self._save_cache()
            return results

        except Exception as e:
            self.logger.error(f"Error fetching Reddit posts: {e}")
            return []

    def compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for a text description."""
        if not self.embedding_available:
            return None
        return self.model.encode(text)

    def update_embedding_cache(self, meme_id: str, description: str):
        """Update embedding cache for a meme description."""
        if not self.embedding_available or meme_id in self.embedding_cache:
            return

        embedding = self.compute_embedding(description)
        if embedding is not None:
            self.embedding_cache[meme_id] = embedding

    def find_most_similar(self, query_embedding: np.ndarray,
                          candidates: List[str],
                          top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find most similar memes based on embedding similarity.

        Args:
            query_embedding: Embedding of the query
            candidates: List of meme IDs to compare against
            top_k: Number of top results to return

        Returns:
            List of (meme_id, similarity_score) tuples
        """
        if not self.embedding_available or not candidates:
            return []

        similarities = []
        for meme_id in candidates:
            if meme_id in self.embedding_cache:
                meme_embedding = self.embedding_cache[meme_id]
                similarity = np.dot(query_embedding, meme_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(meme_embedding)
                )
                similarities.append((meme_id, float(similarity)))

        # Sort by similarity score in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def keyword_match(self, query: str, candidates: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Find memes matching keywords in the query.
        Used as a fallback when embeddings are not available.
        Optimized for r/MemeTemplatesOfficial title format.
        """
        if not candidates:
            return []

        # Normalize query
        query = query.lower()
        query_words = set(query.split())

        # Common words that may not be in titles but describe meme types
        meme_type_words = {"template", "meme", "format", "please", "looking", "for", "need", "want"}

        # Filter out common words that don't help with matching
        filtered_query_words = query_words - meme_type_words

        # If no meaningful words left after filtering, use original words
        if not filtered_query_words and query_words:
            filtered_query_words = query_words

        matches = []

        for meme_id in candidates:
            if meme_id in self.meme_cache:
                meme = self.meme_cache[meme_id]
                description = meme.get('description', '').lower()
                title = meme.get('title', '').lower()

                # r/MemeTemplatesOfficial format often has descriptive titles
                # Count matching words
                all_words = set(description.split()).union(set(title.split()))

                # Check if there are any direct word matches at all
                # If no words match at all, give a very low base score
                matching_words = filtered_query_words.intersection(all_words)
                if not matching_words:
                    # Give a minimal score for ranking purposes
                    base_score = 0.05

                    # Add small popularity and recency boosts
                    upvotes = meme.get('upvotes', 0)
                    popularity_boost = min(0.05, (upvotes / 50000))

                    created_utc = meme.get('created_utc', 0)
                    current_time = time.time()
                    age_in_days = (current_time - created_utc) / 86400
                    recency_boost = min(0.02, max(0, 0.02 - (age_in_days / 365)))

                    # Final score for non-matches is very low
                    final_score = base_score + popularity_boost + recency_boost
                    matches.append((meme_id, final_score))
                    continue

                # Direct matches - words that appear exactly as in the query
                direct_match_score = len(matching_words) / max(1, len(filtered_query_words))

                # Partial matches - for longer phrases and multi-word concepts
                partial_match_score = 0
                if query in title or query in description:
                    partial_match_score = 0.3  # Full phrase match bonus
                else:
                    # Check for key phrase fragments
                    for i in range(2, min(5, len(query.split()) + 1)):  # Check for 2-4 word phrases
                        for j in range(len(query.split()) - i + 1):
                            phrase = " ".join(query.split()[j:j + i])
                            if phrase in title or phrase in description:
                                partial_match_score = max(partial_match_score,
                                                          0.1 * i)  # Longer phrases get higher score

                # Content relevance score
                content_score = direct_match_score + partial_match_score

                # Boost score based on upvotes (popularity indicates community validation)
                upvotes = meme.get('upvotes', 0)
                popularity_boost = min(0.1, (upvotes / 20000))  # Reduce maximum boost to 0.1

                # Recency boost (newer templates might be more relevant)
                created_utc = meme.get('created_utc', 0)
                current_time = time.time()
                age_in_days = (current_time - created_utc) / 86400
                recency_boost = min(0.05, max(0, 0.05 - (age_in_days / 365)))  # Reduce max boost to 0.05

                # Calculate final score - ensure it stays within a reasonable range (0-1)
                final_score = min(1.0, content_score + popularity_boost + recency_boost)

                matches.append((meme_id, final_score))

        # Sort by match score in descending order
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_k]

    def get_meme(self, description: str, top_k: int = 3, refresh: bool = False) -> List[Dict]:
        """
        Get memes matching a description.
        Optimized for r/MemeTemplatesOfficial subreddit structure.

        Args:
            description: Text description of the desired meme
            top_k: Number of top results to return
            refresh: Force refresh of the cache

        Returns:
            List of meme dictionaries
        """
        # Check if we need to refresh the cache
        current_time = time.time()
        cache_age = current_time - self.last_update

        if refresh or cache_age > self.cache_expiry or len(self.meme_cache) < 20:
            self.logger.info("Refreshing meme cache from Reddit")
            # Get a mix of timeframes to ensure diverse templates
            recent_memes = self.fetch_reddit_memes(limit=50, time_filter='month')
            top_memes = self.fetch_reddit_memes(limit=50, time_filter='all')
            self.logger.info(f"Fetched {len(recent_memes)} recent memes and {len(top_memes)} all-time top memes")

        # Prepare search query - optimize for r/MemeTemplatesOfficial
        search_query = description

        # Remove generic words that won't help with searching
        filter_words = ['meme', 'template', 'format', 'please', 'looking', 'for', 'need', 'want']
        cleaned_query = ' '.join([word for word in description.lower().split()
                                  if word not in filter_words])

        if cleaned_query:
            search_query = cleaned_query

        self.logger.info(f"Search query: '{search_query}'")

        candidate_ids = list(self.meme_cache.keys())

        # Find most similar memes
        if self.embedding_available:
            # Compute embedding for the query
            query_embedding = self.compute_embedding(search_query)
            similar_memes = self.find_most_similar(query_embedding, candidate_ids, top_k * 2)
        else:
            # Fall back to keyword matching
            similar_memes = self.keyword_match(search_query, candidate_ids, top_k * 2)

        # For the final ranking, combine similarity with popularity
        results = []
        for meme_id, similarity in similar_memes:
            if meme_id in self.meme_cache:
                meme_data = self.meme_cache[meme_id].copy()  # Create a copy to avoid modifying the cache

                # Calculate final rank score (70% similarity, 30% popularity)
                upvotes = meme_data.get('upvotes', 0)
                normalized_upvotes = min(1.0, upvotes / 10000)  # Normalize upvotes, cap at 10k

                # Small bonus for exact title matches
                title_match_bonus = 0
                if search_query.lower() in meme_data.get('title', '').lower():
                    title_match_bonus = 0.1

                final_score = (similarity * 0.65) + (normalized_upvotes * 0.25) + title_match_bonus
                meme_data['similarity'] = similarity
                meme_data['upvotes'] = upvotes
                meme_data['final_score'] = final_score

                results.append(meme_data)

        # Sort by final score
        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results[:top_k]


class ImgFlipMemeGrabber(MemeGrabberBase):
    """Grabber for meme templates from ImgFlip API."""

    def __init__(self,
                 cache_dir: str = "meme_cache",
                 cache_expiry: int = 7):  # Cache expiry in days
        """
        Initialize the ImgFlipMemeGrabber.

        Args:
            cache_dir: Directory to store cached memes
            cache_expiry: Number of days after which to refresh the cache
        """
        super().__init__(cache_dir)

        self.cache_expiry = cache_expiry * 86400  # Convert days to seconds

        # API endpoints
        self.get_memes_url = "https://api.imgflip.com/get_memes"

        # Cache for meme templates
        self.templates_cache = {}
        self.last_update = 0
        self._load_cache()

    def _load_cache(self):
        """Load meme templates cache from disk if it exists."""
        cache_path = self.cache_dir / "imgflip_templates_cache.pkl"
        last_update_path = self.cache_dir / "imgflip_last_update.txt"

        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    self.templates_cache = pickle.load(f)
                self.logger.info(f"Loaded {len(self.templates_cache)} templates from cache")
            except Exception as e:
                self.logger.error(f"Failed to load templates cache: {e}")

        if last_update_path.exists():
            try:
                with open(last_update_path, 'r') as f:
                    self.last_update = float(f.read().strip())
            except Exception:
                self.last_update = 0

    def _save_cache(self):
        """Save meme templates cache to disk."""
        cache_path = self.cache_dir / "imgflip_templates_cache.pkl"
        last_update_path = self.cache_dir / "imgflip_last_update.txt"

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(self.templates_cache, f)

            with open(last_update_path, 'w') as f:
                f.write(str(time.time()))

            self.last_update = time.time()
            self.logger.info("Cache saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")

    def fetch_all_templates(self) -> List[Dict]:
        """Fetch all available meme templates from ImgFlip API."""
        try:
            response = requests.get(self.get_memes_url)
            data = response.json()

            if data["success"]:
                templates = data["data"]["memes"]

                # Process templates and add to cache
                for template in templates:
                    template_id = template["id"]
                    self.templates_cache[template_id] = {
                        "id": template_id,
                        "name": template["name"],
                        "url": template["url"],
                        "width": template["width"],
                        "height": template["height"],
                        "box_count": template["box_count"],
                        "source": "imgflip"
                    }

                self._save_cache()
                return list(self.templates_cache.values())
            else:
                self.logger.error(f"ImgFlip API error: {data.get('error_message', 'Unknown error')}")
                return []
        except Exception as e:
            self.logger.error(f"Error fetching ImgFlip templates: {e}")
            return []

    def find_template_by_name(self, name: str, refresh: bool = False) -> List[Dict]:
        """
        Find meme templates by name.

        Args:
            name: Name or keyword to search for
            refresh: Force refresh of the cache

        Returns:
            List of matching templates
        """
        # Check if we need to refresh the cache
        current_time = time.time()
        cache_age = current_time - self.last_update

        if refresh or cache_age > self.cache_expiry or len(self.templates_cache) < 5:
            self.logger.info("Refreshing templates cache from ImgFlip")
            self.fetch_all_templates()

        # Normalize search term
        search_term = name.lower()

        # Find matching templates
        matches = []
        for template_id, template in self.templates_cache.items():
            template_name = template["name"].lower()

            # Simple matching mechanism
            if search_term in template_name:
                # Calculate a simple score based on how well the search term matches
                if search_term == template_name:
                    score = 1.0  # Exact match
                elif template_name.startswith(search_term):
                    score = 0.8  # Starts with search term
                else:
                    # Partial match - score based on the relative length of the search term to the template name
                    score = len(search_term) / len(template_name) * 0.6

                match_data = template.copy()
                match_data["score"] = score
                matches.append(match_data)

        # Sort by score in descending order
        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches

    def get_template_info(self, template_id: str) -> Optional[Dict]:
        """
        Get detailed information about a specific template.

        Args:
            template_id: ImgFlip template ID

        Returns:
            Template information dictionary, or None if not found
        """
        # Ensure the template exists
        if template_id not in self.templates_cache:
            self.logger.warning(f"Template ID {template_id} not found in cache")
            # Try to refresh the cache
            self.fetch_all_templates()
            if template_id not in self.templates_cache:
                self.logger.error(f"Template ID {template_id} not found")
                return None

        return self.templates_cache.get(template_id)

    def get_template_by_description(self, description: str, top_k: int = 3) -> List[Dict]:
        """
        Find meme templates that match a description.

        Args:
            description: Description of the meme
            top_k: Number of top results to return

        Returns:
            List of matching templates
        """
        # First, ensure the cache is populated
        if len(self.templates_cache) < 5:
            self.fetch_all_templates()

        # For ImgFlip, we'll use simple keyword matching
        # Convert description to keywords
        keywords = [word.lower() for word in description.split()
                    if len(word) > 3 and word.lower() not in ["meme", "template", "please", "need"]]

        matches = []
        for template_id, template in self.templates_cache.items():
            template_name = template["name"].lower()

            # Count keyword matches
            score = 0
            for keyword in keywords:
                if keyword in template_name:
                    score += 1

            # Only include if there's at least one keyword match
            if score > 0:
                match_data = template.copy()
                match_data["score"] = score / len(keywords)  # Normalize score
                matches.append(match_data)

        # Sort by score in descending order
        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches[:top_k]


class LocalMemeGrabber(MemeGrabberBase):
    """Grabber for memes from local folders organized by reaction categories."""

    def __init__(self, base_folder: str = "reaction_memes"):
        """
        Initialize the LocalMemeGrabber.

        Args:
            base_folder: Path to the base folder containing reaction meme subfolders
        """
        super().__init__()

        self.base_folder = Path(base_folder)
        self.reactions = self._scan_reactions()

    def _scan_reactions(self) -> Dict[str, List[str]]:
        """
        Scan the base folder for reaction categories and meme files.

        Returns:
            Dictionary mapping reaction names to lists of meme file paths
        """
        reactions = {}

        # Create base folder if it doesn't exist
        if not self.base_folder.exists():
            self.base_folder.mkdir(parents=True)
            self.logger.info(f"Created base folder {self.base_folder}")
            return reactions

        # Scan for subfolders (reaction categories)
        for folder in self.base_folder.iterdir():
            if folder.is_dir():
                reaction_name = folder.name
                meme_files = []

                # Scan for image files in the reaction folder
                for file in folder.glob("*.*"):
                    if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
                        meme_files.append(str(file))

                if meme_files:
                    reactions[reaction_name] = meme_files
                    self.logger.info(f"Found {len(meme_files)} memes in category '{reaction_name}'")

        return reactions

    def list_reactions(self) -> List[str]:
        """
        List all available reaction categories.

        Returns:
            List of reaction category names
        """
        return list(self.reactions.keys())

    def get_random_meme(self, reaction: str) -> Optional[str]:
        """
        Get a random meme from a specific reaction category.

        Args:
            reaction: Reaction category name

        Returns:
            File path to a random meme, or None if the category doesn't exist
            or is empty
        """
        # Refresh the reaction categories
        self.reactions = self._scan_reactions()

        if reaction not in self.reactions or not self.reactions[reaction]:
            self.logger.warning(f"Reaction category '{reaction}' not found or empty")
            return None

        # Return a random meme from the category
        return random.choice(self.reactions[reaction])


class MemeGrabber:
    """
    Main class that coordinates between different meme template sources.

    This class provides a unified interface to retrieve meme templates from:
    1. ImgFlip API - For popular, well-known meme templates
    2. Reddit r/MemeTemplatesOfficial - For a wider variety of templates
    3. Local folders - For reaction memes organized by category
    """

    def __init__(self,
                 reddit_client_id: str = None,
                 reddit_client_secret: str = None,
                 reddit_user_agent: str = None,
                 local_memes_folder: str = "reaction_memes",
                 cache_dir: str = "meme_cache"):
        """
        Initialize the MemeGrabber with credentials for various services.

        Args:
            reddit_client_id: Reddit API client ID
            reddit_client_secret: Reddit API client secret
            reddit_user_agent: Reddit API user agent
            local_memes_folder: Path to folder containing local reaction memes
            cache_dir: Directory to store cached memes
        """
        # Initialize each source grabber
        self.reddit_grabber = RedditMemeGrabber(
            reddit_client_id=reddit_client_id,
            reddit_client_secret=reddit_client_secret,
            reddit_user_agent=reddit_user_agent,
            cache_dir=cache_dir
        )

        self.imgflip_grabber = ImgFlipMemeGrabber(
            cache_dir=cache_dir
        )

        self.local_grabber = LocalMemeGrabber(
            base_folder=local_memes_folder
        )

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("MemeGrabber")

    def get_imgflip_templates(self, description: str, limit: int = 3) -> List[Dict]:
        """
        Find meme templates from ImgFlip based on description.

        Args:
            description: Description of the meme
            limit: Maximum number of results to return

        Returns:
            List of matching templates
        """
        return self.imgflip_grabber.get_template_by_description(description, limit)

    def get_reddit_templates(self, description: str, limit: int = 3, refresh: bool = False) -> List[Dict]:
        """
        Find meme templates from Reddit based on description.

        Args:
            description: Description of the meme
            limit: Maximum number of results to return
            refresh: Force refresh of the cache

        Returns:
            List of matching templates
        """
        return self.reddit_grabber.get_meme(description, limit, refresh)

    def get_reaction_meme(self, reaction: str) -> Optional[str]:
        """
        Get a random meme from a specific reaction category.

        Args:
            reaction: Reaction category name

        Returns:
            File path to a random meme, or None if the category doesn't exist
            or is empty
        """
        return self.local_grabber.get_random_meme(reaction)

    def list_reaction_categories(self) -> List[str]:
        """
        List all available reaction categories.

        Returns:
            List of reaction category names
        """
        return self.local_grabber.list_reactions()

    def get_template(self, source: str, query: str, limit: int = 3) -> Dict:
        """
        Get meme templates from the specified source using the query.

        Args:
            source: Source type ('imgflip', 'reddit', 'reaction')
            query: Description for templates or reaction category name
            limit: Maximum number of results to return (not used for 'reaction')

        Returns:
            Dictionary with template data
        """
        if source == "imgflip":
            # Find templates
            templates = self.get_imgflip_templates(query, limit)
            if not templates:
                return {"success": False, "error": "No matching templates found"}

            return {
                "success": True,
                "source": "imgflip",
                "templates": templates
            }

        elif source == "reddit":
            # Find templates
            templates = self.get_reddit_templates(query, limit)
            if not templates:
                return {"success": False, "error": "No matching templates found"}

            return {
                "success": True,
                "source": "reddit",
                "templates": templates
            }

        elif source == "reaction":
            # Get a reaction meme
            meme_path = self.get_reaction_meme(query)
            if not meme_path:
                return {"success": False, "error": f"Reaction category '{query}' not found or empty"}

            return {
                "success": True,
                "source": "reaction",
                "local_path": meme_path,
                "category": query
            }

        else:
            return {"success": False, "error": f"Unknown source: {source}"}


# Example usage
if __name__ == "__main__":
    # Set up MemeGrabber with API credentials
    grabber = MemeGrabber(
        reddit_client_id="YOUR_REDDIT_CLIENT_ID",
        reddit_client_secret="YOUR_REDDIT_CLIENT_SECRET",
        reddit_user_agent="python:meme-grabber:v1.0 (by /u/YOUR_USERNAME)",
        local_memes_folder="reaction_memes"
    )

    # Example 1: Get ImgFlip templates
    imgflip_templates = grabber.get_template(
        source="imgflip",
        query="distracted boyfriend",
        limit=2
    )
    print(f"ImgFlip templates: {json.dumps(imgflip_templates, indent=2)}")

    # Example 2: Get Reddit templates
    reddit_templates = grabber.get_template(
        source="reddit",
        query="guy thinking with math equations",
        limit=2
    )
    print(f"Reddit templates: {json.dumps(reddit_templates, indent=2)}")

    # Example 3: Get a reaction meme
    reaction_meme = grabber.get_template(
        source="reaction",
        query="liar"
    )
    print(f"Reaction meme: {json.dumps(reaction_meme, indent=2)}")

    # Example 4: List all reaction categories
    categories = grabber.list_reaction_categories()
    print(f"Available reaction categories: {categories}")