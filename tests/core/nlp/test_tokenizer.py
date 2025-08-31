"""Tests for tokenization utilities."""

from tenets.core.nlp.tokenizer import CodeTokenizer, TextTokenizer


class TestCodeTokenizer:
    """Test suite for code tokenizer."""

    def test_initialization(self):
        """Test tokenizer initialization."""
        tokenizer = CodeTokenizer(use_stopwords=False)
        assert tokenizer.use_stopwords is False
        assert tokenizer.stopwords is None

        tokenizer_with_stopwords = CodeTokenizer(use_stopwords=True)
        assert tokenizer_with_stopwords.use_stopwords is True
        assert tokenizer_with_stopwords.stopwords is not None

    def test_tokenize_basic(self):
        """Test basic tokenization."""
        tokenizer = CodeTokenizer()

        code = "def hello_world():\n    print('Hello')"
        tokens = tokenizer.tokenize(code)

        assert isinstance(tokens, list)
        assert "def" in tokens
        assert "hello" in tokens
        assert "world" in tokens
        assert "print" in tokens

    def test_tokenize_camel_case(self):
        """Test camelCase tokenization."""
        tokenizer = CodeTokenizer()

        tokens = tokenizer.tokenize("getUserName")

        assert "get" in tokens
        assert "user" in tokens
        assert "name" in tokens
        assert "getusername" in tokens  # Original preserved

    def test_tokenize_snake_case(self):
        """Test snake_case tokenization."""
        tokenizer = CodeTokenizer()

        tokens = tokenizer.tokenize("process_user_data")

        assert "process" in tokens
        assert "user" in tokens
        assert "data" in tokens

    def test_tokenize_identifier(self):
        """Test tokenizing single identifier."""
        tokenizer = CodeTokenizer()

        # CamelCase
        tokens = tokenizer.tokenize_identifier("MyClassName")
        assert "my" in tokens
        assert "class" in tokens
        assert "name" in tokens

        # snake_case
        tokens = tokenizer.tokenize_identifier("my_function_name")
        assert "my" in tokens
        assert "function" in tokens
        assert "name" in tokens

        # SCREAMING_SNAKE_CASE
        tokens = tokenizer.tokenize_identifier("MAX_BUFFER_SIZE")
        assert "max" in tokens
        assert "buffer" in tokens
        assert "size" in tokens

    def test_tokenize_with_language(self):
        """Test language-specific tokenization."""
        tokenizer = CodeTokenizer()

        python_code = "import numpy as np\nclass DataProcessor:\n    pass"
        tokens = tokenizer.tokenize(python_code, language="python")

        assert "import" in tokens
        assert "numpy" in tokens
        assert "data" in tokens
        assert "processor" in tokens

    def test_tokenize_empty(self):
        """Test tokenizing empty text."""
        tokenizer = CodeTokenizer()

        assert tokenizer.tokenize("") == []
        assert tokenizer.tokenize("   ") == []

    def test_stopword_filtering(self):
        """Test stopword filtering."""
        tokenizer = CodeTokenizer(use_stopwords=True)

        text = "the function is a simple one"
        tokens = tokenizer.tokenize(text)

        # Common stopwords should be filtered
        assert "the" not in tokens or len(tokens) < 6  # Some stopwords removed
        assert "function" in tokens
        assert "simple" in tokens


class TestTextTokenizer:
    """Test suite for text tokenizer."""

    def test_initialization(self):
        """Test text tokenizer initialization."""
        tokenizer = TextTokenizer(use_stopwords=True)
        assert tokenizer.use_stopwords is True
        assert tokenizer.stopwords is not None

        tokenizer_no_stop = TextTokenizer(use_stopwords=False)
        assert tokenizer_no_stop.use_stopwords is False
        assert tokenizer_no_stop.stopwords is None

    def test_tokenize_basic(self):
        """Test basic text tokenization."""
        tokenizer = TextTokenizer(use_stopwords=False)

        text = "This is a simple test sentence."
        tokens = tokenizer.tokenize(text)

        assert isinstance(tokens, list)
        assert "this" in tokens
        assert "simple" in tokens
        assert "test" in tokens
        assert "sentence" in tokens

    def test_tokenize_min_length(self):
        """Test minimum token length."""
        tokenizer = TextTokenizer(use_stopwords=False)

        text = "I am a test"
        tokens = tokenizer.tokenize(text, min_length=2)

        # Single character words should be filtered
        assert "i" not in [t.lower() for t in tokens]
        assert "am" in tokens
        assert "test" in tokens

    def test_extract_ngrams(self):
        """Test n-gram extraction."""
        tokenizer = TextTokenizer(use_stopwords=False)

        text = "machine learning algorithms"

        # Bigrams
        bigrams = tokenizer.extract_ngrams(text, n=2)
        assert "machine learning" in bigrams
        assert "learning algorithms" in bigrams

        # Trigrams
        trigrams = tokenizer.extract_ngrams(text, n=3)
        assert "machine learning algorithms" in trigrams

    def test_extract_ngrams_short_text(self):
        """Test n-gram extraction with short text."""
        tokenizer = TextTokenizer()

        text = "hello"
        bigrams = tokenizer.extract_ngrams(text, n=2)
        assert bigrams == []

    def test_stopword_filtering(self):
        """Test stopword filtering in text."""
        tokenizer = TextTokenizer(use_stopwords=True)

        text = "The quick brown fox jumps over the lazy dog"
        tokens = tokenizer.tokenize(text)

        # Common stopwords should be filtered
        assert "the" not in [t.lower() for t in tokens]
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens

    def test_tokenize_empty_text(self):
        """Test tokenizing empty text."""
        tokenizer = TextTokenizer()

        assert tokenizer.tokenize("") == []
        assert tokenizer.tokenize("   ") == []
        assert tokenizer.extract_ngrams("", n=2) == []
