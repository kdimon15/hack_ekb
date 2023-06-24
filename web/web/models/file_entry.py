from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class FileEntry:
    label: str
    parse_dates: List[str]
    df: Optional[pd.DataFrame] = None

    @property
    def is_uploaded(self) -> bool:
        return self.df is not None
