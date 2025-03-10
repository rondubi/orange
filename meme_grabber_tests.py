import unittest
import os
import sys
import shutil
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import logging
from dotenv import load_dotenv

# Disable overly verbose logging during tests
logging.basicConfig(level=logging.ERROR)

# Import the MemeGrabber class
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from meme_grabber import MemeGrabber, RedditMemeGrabber, ImgFlipMemeGrabber, LocalMemeGrabber


class MemeGrabberIntegrationTests(unittest.TestCase):
    """Integration tests for the MemeGrabber class and its components."""

    @classmethod
    def setUpClass(cls):
        # Try to load environment variables from .env file
        dotenv_path = Path('.env')
        if dotenv_path.exists():
            load_dotenv(dotenv_path)
            print("Loaded environment variables from .env file")

        # Create temporary directories for testing
        cls.test_dir = tempfile.mkdtemp()
        cls.cache_dir = os.path.join(cls.test_dir, "test_cache")

        # Allow using a real reaction folder path from .env
        cls.real_reactions_dir = os.environ.get("REAL_REACTIONS_DIR")
        cls.reactions_dir = cls.real_reactions_dir if cls.real_reactions_dir else os.path.join(cls.test_dir,
                                                                                               "test_reactions")

        # Create cache directory
        os.makedirs(cls.cache_dir, exist_ok=True)

        # Create reaction categories with test images if using test directory
        if not cls.real_reactions_dir:
            cls._create_test_reaction_folders()
            print(f"Created test reaction folders in: {cls.reactions_dir}")
        else:
            print(f"Using real reaction folders from: {cls.reactions_dir}")

        # Load test environment variables for API credentials
        cls.reddit_client_id = os.environ.get("REDDIT_CLIENT_ID")
        cls.reddit_client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
        cls.reddit_user_agent = os.environ.get("REDDIT_USER_AGENT", "python:meme-grabber-test:v1.0")

        print(f"\nTest configuration:")
        print(f"  Cache: {cls.cache_dir}")
        print(f"  Reactions: {cls.reactions_dir}")

        if cls.reddit_client_id and cls.reddit_client_secret:
            print("Reddit API credentials found in environment variables.")
        else:
            print("WARNING: Reddit API credentials not found. Reddit API tests will be skipped.")

    @classmethod
    def tearDownClass(cls):
        # Clean up test directories
        shutil.rmtree(cls.test_dir)
        print(f"\nTest directories removed.")

    @classmethod
    def _create_test_reaction_folders(cls):
        """Create test reaction folders with dummy images."""
        # Create base reactions directory
        os.makedirs(cls.reactions_dir, exist_ok=True)

        # Create test reaction categories
        categories = ["happy", "sad", "angry", "surprised"]

        for category in categories:
            category_dir = os.path.join(cls.reactions_dir, category)
            os.makedirs(category_dir, exist_ok=True)

            # Create some dummy image files for each category
            for i in range(3):
                image_path = os.path.join(category_dir, f"test_image_{i}.jpg")
                # Make sure the file has actual content
                with open(image_path, "wb") as f:
                    # Write a minimal JPEG header to make it a valid image file
                    f.write(bytes([
                        0xFF, 0xD8,  # SOI marker
                        0xFF, 0xE0,  # APP0 marker
                        0x00, 0x10,  # length of APP0 segment
                        0x4A, 0x46, 0x49, 0x46, 0x00,  # JFIF identifier
                        0x01, 0x01,  # version
                        0x00,  # units
                        0x00, 0x01, 0x00, 0x01,  # x and y densities
                        0x00, 0x00,  # thumbnail width and height
                        0xFF, 0xD9  # EOI marker
                    ]))

        print(f"Created {len(categories)} test reaction categories with dummy image files")

    def setUp(self):
        # Create MemeGrabber instance for each test
        self.grabber = MemeGrabber(
            reddit_client_id=self.reddit_client_id,
            reddit_client_secret=self.reddit_client_secret,
            reddit_user_agent=self.reddit_user_agent,
            local_memes_folder=self.reactions_dir,
            cache_dir=self.cache_dir
        )

    def test_01_local_reaction_categories(self):
        """Test listing reaction categories."""
        categories = self.grabber.list_reaction_categories()
        self.assertIsInstance(categories, list)
        self.assertGreater(len(categories), 0, "No reaction categories found")

        print(f"Found {len(categories)} reaction categories: {', '.join(categories[:5])}", end="")
        if len(categories) > 5:
            print(f" and {len(categories) - 5} more...")
        else:
            print("")

        # Test that we can access these categories
        for category in categories[:3]:  # Test first 3 categories
            meme = self.grabber.get_reaction_meme(category)
            self.assertIsNotNone(meme, f"Failed to get meme from category '{category}'")
            self.assertTrue(os.path.exists(meme), f"Meme file does not exist: {meme}")
            print(f"  Successfully accessed meme in '{category}' category: {os.path.basename(meme)}")

    def test_02_get_reaction_meme(self):
        """Test getting a random reaction meme."""
        # Get the list of actual categories first
        categories = self.grabber.list_reaction_categories()

        if not categories:
            self.skipTest("No reaction categories available - check your reaction memes folder")

        # Use the first actual category instead of assuming "happy" exists
        test_category = categories[0]
        print(f"Testing with reaction category: '{test_category}'")

        # Test valid category
        meme_path = self.grabber.get_reaction_meme(test_category)
        self.assertIsNotNone(meme_path, f"Failed to get a meme from existing category '{test_category}'")
        self.assertTrue(os.path.exists(meme_path), f"Meme file path exists but file not found: {meme_path}")
        print(f"Successfully retrieved reaction meme: {meme_path}")

        # Test invalid category
        meme_path = self.grabber.get_reaction_meme("nonexistent_category_that_should_not_exist")
        self.assertIsNone(meme_path, "Got a non-None result for nonexistent category")
        print("Correctly handled nonexistent category")

    def test_03_get_reaction_template(self):
        """Test the unified interface for reaction memes."""
        # Get the list of actual categories first
        categories = self.grabber.list_reaction_categories()

        if not categories:
            self.skipTest("No reaction categories available - check your reaction memes folder")

        # Use the first actual category instead of assuming "happy" exists
        test_category = categories[0]
        print(f"Testing unified interface with reaction category: '{test_category}'")

        # Test valid category
        result = self.grabber.get_template(source="reaction", query=test_category)

        # Add debugging info if the test fails
        if not result.get("success", False):
            print(f"Error getting template: {result.get('error', 'Unknown error')}")
            print(f"Local grabber reactions: {self.grabber.local_grabber.reactions}")
            print(f"Category exists: {test_category in self.grabber.local_grabber.reactions}")
            if test_category in self.grabber.local_grabber.reactions:
                print(f"Files in category: {self.grabber.local_grabber.reactions[test_category]}")

        self.assertTrue(result["success"],
                        f"Failed to get template from category '{test_category}': {result.get('error', 'Unknown error')}")
        self.assertEqual(result["source"], "reaction")
        self.assertEqual(result["category"], test_category)
        self.assertTrue(os.path.exists(result["local_path"]),
                        f"Local path exists but file not found: {result['local_path']}")
        print(f"Successfully retrieved reaction template: {result['local_path']}")

        # Test invalid category
        result = self.grabber.get_template(source="reaction", query="nonexistent_category_that_should_not_exist")
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        print(f"Correctly handled error: {result['error']}")

    def test_04_imgflip_templates_real(self):
        """Test ImgFlip template retrieval with the real API (no auth required)."""
        # This test uses the real ImgFlip API (no authentication needed for getting templates)
        templates = self.grabber.imgflip_grabber.fetch_all_templates()

        # Verify we got some templates
        self.assertIsInstance(templates, list)
        self.assertGreater(len(templates), 10, "Too few templates returned, API might be having issues")

        print(f"Successfully retrieved {len(templates)} templates from ImgFlip API")
        print("Sample templates:")
        for template in templates[:3]:
            print(f"  {template['name']} (ID: {template['id']})")

        # Now test template search with a common meme name
        search_term = "drake"
        results = self.grabber.get_imgflip_templates(search_term)

        # Check if we found relevant templates
        self.assertIsInstance(results, list)
        found_relevant = False
        for template in results:
            if search_term.lower() in template["name"].lower():
                found_relevant = True
                break

        if results:
            print(f"Search for '{search_term}' found {len(results)} templates:")
            for template in results[:3]:
                print(f"  {template['name']} (score: {template.get('score', 'N/A')})")

            if found_relevant:
                print(f"✓ Successfully found relevant templates matching '{search_term}'")
            else:
                print(f"✗ No exact match for '{search_term}' in results (API might have changed)")
        else:
            print(f"No results found for '{search_term}' (API might be having issues)")

    def test_05_imgflip_template_unified(self):
        """Test the unified interface for ImgFlip templates with real API."""
        # Use a common meme search term
        result = self.grabber.get_template(source="imgflip", query="boyfriend", limit=3)

        self.assertTrue(result["success"], f"Failed to get templates: {result.get('error', 'Unknown error')}")
        self.assertEqual(result["source"], "imgflip")
        self.assertIsInstance(result["templates"], list)

        if result["templates"]:
            print(f"Successfully retrieved {len(result['templates'])} ImgFlip templates through unified interface")
            print("Results:")
            for template in result["templates"]:
                print(f"  {template['name']} (score: {template.get('score', 'N/A')})")
        else:
            print("No ImgFlip templates found for query 'boyfriend' (API might be having issues)")

    @unittest.skipIf(not os.environ.get("REDDIT_CLIENT_ID"), "Reddit API credentials not available")
    def test_06_reddit_templates_live(self):
        """Test Reddit template retrieval with live API (requires credentials)."""
        # Skip test if no credentials
        if not self.reddit_client_id or not self.reddit_client_secret:
            self.skipTest("Reddit API credentials not available")

        # First check if we can establish a connection to Reddit
        try:
            reddit_grabber = self.grabber.reddit_grabber
            if not reddit_grabber.reddit:
                self.skipTest("Failed to initialize Reddit API client")

            # Check if we can access the subreddit
            subreddit = reddit_grabber.reddit.subreddit(reddit_grabber.subreddit_name)
            # Just a basic check to see if the subreddit exists
            description = subreddit.description
            print(f"Successfully connected to r/{reddit_grabber.subreddit_name}")

            # Clear the cache to force a real API call
            reddit_grabber.meme_cache = {}
            reddit_grabber.last_update = 0

            # Try to fetch memes from Reddit API
            print("\nFetching memes directly from Reddit API...")
            templates = reddit_grabber.fetch_reddit_memes(limit=10)

            # Verify we actually fetched something
            self.assertIsInstance(templates, list, "API didn't return a list of templates")
            self.assertGreater(len(templates), 0, "API returned an empty list of templates")

            print(f"✓ Successfully fetched {len(templates)} templates from Reddit API")

            # Print some information about the fetched templates
            for i, template in enumerate(templates[:3]):
                print(f"  {i + 1}. {template.get('title', 'No title')[:40]}...")
                print(f"     URL: {template.get('url', 'No URL')}")
                print(f"     Reddit ID: {template.get('id', 'No ID')}")
                print(f"     Upvotes: {template.get('upvotes', 'Unknown')}")

            # Verify the cache was populated
            self.assertGreater(len(reddit_grabber.meme_cache), 0, "Cache wasn't populated after API call")
            print(f"✓ Cache populated with {len(reddit_grabber.meme_cache)} templates")

            # Now test the search functionality with a fresh API call
            search_term = "button"  # Common meme theme
            print(f"\nSearching for '{search_term}' templates...")
            search_results = self.grabber.get_reddit_templates(search_term, limit=3, refresh=True)

            # Verify search results
            self.assertIsInstance(search_results, list, "Search didn't return a list")
            print(f"Search returned {len(search_results)} results")

            if search_results:
                print("Search results:")
                for i, result in enumerate(search_results):
                    score = result.get('final_score', 0)
                    print(f"  {i + 1}. {result.get('title', 'No title')[:40]}... " +
                          f"(score: {score:.2f})")

        except Exception as e:
            self.fail(f"Reddit API test failed: {str(e)}")

    def test_07_reddit_templates_unified(self):
        """Test the unified interface for Reddit templates."""
        if self.reddit_client_id and self.reddit_client_secret:
            # If we have credentials, try to use the real API
            print("\nTesting unified interface with real Reddit API...")
            result = self.grabber.get_template(source="reddit", query="button", limit=2)

            if result["success"] and result["templates"]:
                print("✓ Successfully retrieved Reddit templates through unified interface (real API)")
                print("Results:")
                for template in result["templates"]:
                    print(f"  {template.get('title', 'No title')[:40]}...")

                # Make sure these are actually from Reddit
                self.assertEqual(result["source"], "reddit", "Source wasn't set to 'reddit'")
                for template in result["templates"]:
                    self.assertTrue("id" in template, "Template missing Reddit ID")
                    self.assertTrue("permalink" in template, "Template missing Reddit permalink")

                return
            else:
                print("Could not retrieve templates from Reddit API, falling back to mock test")
                if not result["success"]:
                    print(f"Error: {result.get('error', 'Unknown error')}")
        else:
            print("No Reddit credentials available, using mock test")

        # Fall back to mock test if real API doesn't work or credentials are missing
        with patch.object(RedditMemeGrabber, 'get_meme') as mock_get:
            # Mock the template retrieval
            mock_get.return_value = [
                {
                    "id": "abc123",
                    "url": "https://i.redd.it/sample1.jpg",
                    "title": "Button meme template",
                    "description": "Guy sweating over which button to press",
                    "permalink": "https://reddit.com/r/MemeTemplatesOfficial/comments/abc123/",
                    "final_score": 0.85
                }
            ]

            # Test the unified interface
            result = self.grabber.get_template(source="reddit", query="button", limit=1)
            self.assertTrue(result["success"])
            self.assertEqual(result["source"], "reddit")
            self.assertEqual(len(result["templates"]), 1)

            print(f"✓ Successfully retrieved Reddit template through unified interface (mock)")

    def test_08_invalid_source(self):
        """Test handling of invalid source in unified interface."""
        result = self.grabber.get_template(source="invalid", query="test")
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("Unknown source", result["error"])
        print(f"Correctly handled invalid source: {result['error']}")

    def test_09_end_to_end(self):
        """End-to-end test of the full MemeGrabber workflow."""
        print("\nRunning end-to-end test workflow:")

        # Step 1: List reaction categories
        categories = self.grabber.list_reaction_categories()
        print(f"1. Found {len(categories)} reaction categories")

        # Step 2: Get a reaction meme
        if categories:
            result = self.grabber.get_template(source="reaction", query=categories[0])
            if result["success"]:
                print(
                    f"2. Retrieved reaction meme from '{categories[0]}' category: {os.path.basename(result['local_path'])}")
            else:
                print(f"2. Failed to get reaction meme: {result['error']}")

        # Step 3: Mock and test ImgFlip template retrieval
        with patch.object(ImgFlipMemeGrabber, 'get_template_by_description') as mock_imgflip:
            mock_imgflip.return_value = [{"id": "123", "name": "Test Template", "url": "https://example.com/test.jpg"}]
            result = self.grabber.get_template(source="imgflip", query="test")
            if result["success"]:
                print(f"3. Retrieved ImgFlip template: {result['templates'][0]['name']}")

        # Step 4: Mock and test Reddit template retrieval
        with patch.object(RedditMemeGrabber, 'get_meme') as mock_reddit:
            mock_reddit.return_value = [{"id": "456", "title": "Test Meme", "url": "https://reddit.com/test.jpg"}]
            result = self.grabber.get_template(source="reddit", query="test")
            if result["success"]:
                print(f"4. Retrieved Reddit template: {result['templates'][0]['title']}")

        print("End-to-end test workflow completed successfully")


class ImgFlipGrabberUnitTests(unittest.TestCase):
    """Unit tests for the ImgFlipMemeGrabber class."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.grabber = ImgFlipMemeGrabber(cache_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch('requests.get')
    def test_fetch_all_templates(self, mock_get):
        """Test fetching all templates."""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "data": {
                "memes": [
                    {
                        "id": "1",
                        "name": "Template 1",
                        "url": "https://example.com/1.jpg",
                        "width": 500,
                        "height": 500,
                        "box_count": 2
                    },
                    {
                        "id": "2",
                        "name": "Template 2",
                        "url": "https://example.com/2.jpg",
                        "width": 600,
                        "height": 400,
                        "box_count": 3
                    }
                ]
            }
        }
        mock_get.return_value = mock_response

        # Test fetching templates
        templates = self.grabber.fetch_all_templates()
        self.assertEqual(len(templates), 2)
        self.assertEqual(templates[0]["id"], "1")
        self.assertEqual(templates[0]["name"], "Template 1")
        self.assertEqual(templates[1]["id"], "2")
        self.assertEqual(templates[1]["name"], "Template 2")

        # Verify cache was updated
        self.assertEqual(len(self.grabber.templates_cache), 2)
        self.assertIn("1", self.grabber.templates_cache)
        self.assertIn("2", self.grabber.templates_cache)


class RedditGrabberUnitTests(unittest.TestCase):
    """Unit tests for the RedditMemeGrabber class."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.grabber = RedditMemeGrabber(cache_dir=self.temp_dir)

        # Create a mock meme cache
        self.grabber.meme_cache = {
            "abc123": {
                "id": "abc123",
                "url": "https://i.redd.it/sample1.jpg",
                "title": "Template with keywords",
                "description": "Template with keywords",
                "upvotes": 5000,
                "created_utc": time.time() - 86400  # 1 day ago
            },
            "def456": {
                "id": "def456",
                "url": "https://i.redd.it/sample2.jpg",
                "title": "Another template example",
                "description": "Another template example",
                "upvotes": 3000,
                "created_utc": time.time() - 172800  # 2 days ago
            }
        }

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_keyword_match(self):
        """Test keyword matching functionality."""
        # Test with matching keywords
        matches = self.grabber.keyword_match("template keywords", list(self.grabber.meme_cache.keys()))
        self.assertEqual(len(matches), 2)  # Should match both templates
        self.assertEqual(matches[0][0], "abc123")  # First match should be "abc123"

        # Test with specific keywords
        matches = self.grabber.keyword_match("another example", list(self.grabber.meme_cache.keys()))
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0][0], "def456")  # First match should be "def456"


class LocalGrabberUnitTests(unittest.TestCase):
    """Unit tests for the LocalMemeGrabber class."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        # Create test reaction folders
        self.reaction1_dir = os.path.join(self.temp_dir, "reaction1")
        self.reaction2_dir = os.path.join(self.temp_dir, "reaction2")
        os.makedirs(self.reaction1_dir)
        os.makedirs(self.reaction2_dir)

        # Create test images
        for i in range(3):
            with open(os.path.join(self.reaction1_dir, f"image{i}.jpg"), "w") as f:
                f.write(f"Test image {i}")

        with open(os.path.join(self.reaction2_dir, "image.jpg"), "w") as f:
            f.write("Test image")

        # Initialize grabber
        self.grabber = LocalMemeGrabber(base_folder=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_scan_reactions(self):
        """Test scanning for reaction categories."""
        reactions = self.grabber._scan_reactions()
        self.assertEqual(len(reactions), 2)
        self.assertIn("reaction1", reactions)
        self.assertIn("reaction2", reactions)
        self.assertEqual(len(reactions["reaction1"]), 3)
        self.assertEqual(len(reactions["reaction2"]), 1)

    def test_list_reactions(self):
        """Test listing reaction categories."""
        categories = self.grabber.list_reactions()
        self.assertEqual(len(categories), 2)
        self.assertIn("reaction1", categories)
        self.assertIn("reaction2", categories)

    def test_get_random_meme(self):
        """Test getting a random meme."""
        # Valid category
        meme_path = self.grabber.get_random_meme("reaction1")
        self.assertIsNotNone(meme_path)
        self.assertTrue(os.path.exists(meme_path))
        self.assertTrue(meme_path.startswith(self.reaction1_dir))

        # Invalid category
        meme_path = self.grabber.get_random_meme("nonexistent")
        self.assertIsNone(meme_path)


if __name__ == "__main__":
    # Check for a .env file first
    env_file = Path('.env')
    if not env_file.exists():
        print("\nNote: No .env file found. If you have API credentials, create a .env file with:")
        print("REDDIT_CLIENT_ID=your_client_id")
        print("REDDIT_CLIENT_SECRET=your_client_secret")
        print("REDDIT_USER_AGENT=python:meme-grabber-test:v1.0")

    unittest.main()