from dataclasses import dataclass

from models.file_entry import FileEntry


@dataclass
class Files:
    cargo = FileEntry("Грузоперевозки", parse_dates=[])
    construction = FileEntry("Данные рынка стройматериалов", parse_dates=[])
    lme_index = FileEntry("Индекс LME", parse_dates=[])
    macro = FileEntry("Макропоказатели", parse_dates=[])
    metal_market = FileEntry("Показатели рынка металла", parse_dates=[])
    fuel = FileEntry("Топливо", parse_dates=[])
    raw_prices = FileEntry("Цены на сырье", parse_dates=[])
    chmf = FileEntry("Акции CHMF", parse_dates=["Date"])
    magn = FileEntry("Акции MAGN", parse_dates=["Дата"])
    nlmk = FileEntry("Акции NLMK", parse_dates=["Date"])
    test = FileEntry("Тестовый датасет", parse_dates=[])
    train = FileEntry("Тренировочный датасет", parse_dates=[])

    @property
    def is_uploaded(self) -> bool:
        return (
            self.cargo.is_uploaded
            and self.construction.is_uploaded
            and self.lme_index.is_uploaded
            and self.macro.is_uploaded
            and self.metal_market.is_uploaded
            and self.fuel.is_uploaded
            and self.raw_prices.is_uploaded
            and self.chmf.is_uploaded
            and self.nlmk.is_uploaded
            and self.test.is_uploaded
            and self.train.is_uploaded
        )
