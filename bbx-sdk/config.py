# Copyright 2025 Ilya Makarov, Krasnoyarsk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    BLACKBOX_API_URL: str = "http://localhost:8000"
    BLACKBOX_USERNAME: str = ""
    BLACKBOX_PASSWORD: str = ""
    BLACKBOX_API_TIMEOUT: int = 30
    LOG_LEVEL: str = "INFO"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
