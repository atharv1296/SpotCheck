"""
Django settings for Forbes Marshall Parking System.
"""

from pathlib import Path
import os
from typing import Optional

# Optional: load environment variables from a local .env file if present
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    # dotenv is optional; if not installed, environment variables must be set externally
    pass

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-forbesmarshall-parking-system-development-key'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['localhost', '127.0.0.1']

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'parking_app',
    'driver_applications',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'core.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'core.wsgi.application'

# Database - Oracle Configuration (reads from environment)
# Supported env vars:
#   ORACLE_DSN (preferred) e.g. "localhost:1521/orclpdb"
#   ORACLE_HOST, ORACLE_PORT, ORACLE_SERVICE_NAME (used to build DSN if ORACLE_DSN not set)
#   ORACLE_USER, ORACLE_PASSWORD
#   ORACLE_THICK_MODE=1 and ORACLE_CLIENT_LIB_DIR=C:\oracle\instantclient_19_23 (optional; only if using thick mode)

def _oracle_dsn() -> str:
    dsn = os.getenv('ORACLE_DSN')
    if dsn:
        return dsn
    host = os.getenv('ORACLE_HOST', 'localhost')
    port = os.getenv('ORACLE_PORT', '1521')
    service = os.getenv('ORACLE_SERVICE_NAME', 'orclpdb')
    return f"{host}:{port}/{service}"

# Try to initialize thick mode if explicitly requested via env var
if os.getenv('ORACLE_THICK_MODE') in ('1', 'true', 'True', 'yes', 'on'):
    try:
        import oracledb  # type: ignore
        lib_dir: Optional[str] = os.getenv('ORACLE_CLIENT_LIB_DIR')
        if lib_dir:
            oracledb.init_oracle_client(lib_dir=lib_dir)
    except Exception:
        # If init fails, the oracledb driver will run in thin mode (no Instant Client)
        pass

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.oracle',
        'NAME': _oracle_dsn(),
        'USER': os.getenv('ORACLE_USER', 'system'),
        'PASSWORD': os.getenv('ORACLE_PASSWORD', ''),
        'OPTIONS': {
            # Add driver options if needed, e.g., encoding, arraysize tuning, etc.
            # 'encoding': 'UTF-8',
        },
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'Asia/Kolkata'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATICFILES_DIRS = [
    BASE_DIR / 'static',
]
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Logging
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': BASE_DIR / 'dashboard.log',
            'formatter': 'verbose',
        },
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
        'parking_file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': BASE_DIR / 'parking_activity.log',
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': True,
        },
        'parking_app': {
            'handlers': ['parking_file', 'console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}

# Caching
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'forbes-marshall-parking',
        'TIMEOUT': 300,  # 5 minutes
        'OPTIONS': {
            'MAX_ENTRIES': 1000,
            'CULL_FREQUENCY': 3,
        }
    }
}

# Session configuration
SESSION_COOKIE_AGE = 86400  # 24 hours
SESSION_SAVE_EVERY_REQUEST = True
SESSION_EXPIRE_AT_BROWSER_CLOSE = True

# Security settings for production
if not DEBUG:
    SECURE_BROWSER_XSS_FILTER = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
    X_FRAME_OPTIONS = 'DENY'
    SECURE_HSTS_SECONDS = 86400
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True

# Custom settings for Forbes Marshall Parking System
PARKING_SYSTEM_CONFIG = {
    'COMPANY_NAME': 'Forbes Marshall',
    'SYSTEM_NAME': 'SpotCheck',
    'VERSION': '2.0.0',
    'AUTO_REFRESH_INTERVAL': 30,  # seconds
    'MAX_PARKING_DURATION': 24,  # hours
    'TIMEZONE': 'Asia/Kolkata',
    'CURRENCY': 'INR',
    'PARKING_RATES': {
        'hourly': 50,
        'daily': 400,
        'monthly': 8000,
    },
    'FINE_AMOUNTS': {
        'unauthorized_parking': 500,
        'overtime': 100,
        'wrong_slot_type': 200,
    }
}

# Authentication settings (to be integrated with main website login later)
# LOGIN_URL = '/accounts/login/'
# LOGIN_REDIRECT_URL = '/'
# LOGOUT_REDIRECT_URL = '/'
