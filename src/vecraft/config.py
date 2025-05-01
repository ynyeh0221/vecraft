from pydantic import BaseSettings

class Settings(BaseSettings):
    data_path: str = "data/"
    page_size: int = 4096
    default_index: str = "brute_force"
    storage_backend: str = "file_mmap"

    class Config:
        env_prefix = "VEcDB_"

settings = Settings()