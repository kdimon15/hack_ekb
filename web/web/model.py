import pickle
from datetime import date
from typing import Tuple

import numpy as np
import pandas as pd
from models.files import Files


class DateIsNotPresentException(Exception):
    pass


class Model:
    def __init__(self, path: str) -> None:
        self.path = path
        self._load_model()

    def _load_model(self):
        self.models, self.month_stats = pickle.load(open(self.path, "rb"))

    def make_prediction_for_date(
        self,
        files: Files,
        date_to: date,
    ) -> int:
        assert files.train.df is not None and files.test.df is not None

        all_df = pd.concat((files.train.df, files.test.df)).reset_index(drop=True)
        all_df = self._add_info(all_df, files)
        all_df["Изменение в месяце"] = all_df["month"].apply(
            lambda x: self.month_stats[x]
        )

        index = all_df[
            (all_df.dt.dt.year == date_to.year)
            & (all_df.dt.dt.month == date_to.month)
            & (all_df.dt.dt.day == date_to.day)
        ]

        if len(index) == 0:
            raise DateIsNotPresentException()

        index = index.index[0]
        cur_df = self._make_features(all_df.iloc[: index + 1])
        pred = [x[0].predict_proba(cur_df[x[1]])[1] for x in self.models]
        p = 1
        for x in pred:
            if x > 0.5:
                p += 1
            else:
                break
        return p

    def predict(self, files: Files) -> Tuple[int, pd.DataFrame]:
        assert files.train.df is not None and files.test.df is not None

        all_df = pd.concat((files.train.df, files.test.df)).reset_index(drop=True)
        all_df = self._add_info(all_df, files)
        all_df["Изменение в месяце"] = all_df["month"].apply(
            lambda x: self.month_stats[x]
        )

        test_data = []
        for i in range(len(files.train.df), len(all_df)):
            cur_df = all_df.iloc[:i]
            test_data.append(self._make_features(cur_df))

        all_preds = []
        for data in test_data:
            cur_preds = [x[0].predict_proba(data[x[1]])[1] for x in self.models]
            all_preds.append(cur_preds)
        all_preds = np.array(all_preds)

        test_preds = []
        for data in test_data:
            cur_preds = [x[0].predict_proba(data[x[1]])[1] for x in self.models]
            pred = 1
            for c, x in enumerate(cur_preds):
                if x > 0.36:
                    pred += 1
                else:
                    break
            test_preds.append(pred)

        final_preds = []
        next = 0
        for id, x in enumerate(test_preds):
            if id == next:
                final_preds.append(min(x, len(test_preds) - len(final_preds)))
                for _ in range(x - 1):
                    if len(final_preds) < len(files.test.df):
                        final_preds.append(0)
                next = id + x

        files.test.df["Объем"] = final_preds

        tmp = []
        for ob, price in zip(files.test.df["Объем"], files.test.df["Цена на арматуру"]):
            for _ in range(ob):
                tmp.append(price)

        prices_df = pd.DataFrame(
            {"Стоимость": files.test.df["Цена на арматуру"], "Дата": files.test.df.dt},
        )
        prices_df["Тип"] = "Рынок"

        res_df = pd.DataFrame({"Стоимость": tmp, "Дата": files.test.df.dt})
        res_df["Тип"] = "Предсказание"
        return (
            self._compute_metric(files.test.df),
            pd.concat((prices_df, res_df)),
        )

    def _compute_metric(self, df: pd.DataFrame):
        df = df.set_index("dt")
        tender_price = df["Цена на арматуру"]
        decision = df["Объем"]
        start_date = df.index.min()
        end_date = df.index.max()
        _results = []
        _active_weeks = 0
        for report_date in pd.date_range(start_date, end_date, freq="W-MON"):
            _fixed_price = 0
            if _active_weeks == 0:  # Пришла пора нового тендера
                _fixed_price = tender_price.loc[report_date]
                _active_weeks = int(decision.loc[report_date])
            _results.append(_fixed_price)
            _active_weeks += -1
        cost = sum(_results)
        return cost

    def _add_info(self, all_df: pd.DataFrame, files: Files):
        if files.raw_prices.df is not None:
            files.raw_prices.df = files.raw_prices.df[
                files.raw_prices.df.dt.isin(all_df.dt)
            ].dropna(axis=1)
            files.raw_prices.df.columns = [
                f"сырье_{x}" if files.raw_prices.df.columns[x] != "dt" else "dt"
                for x in range(len(files.raw_prices.df.columns))
            ]
            all_df = all_df.merge(files.raw_prices.df, how="left", on="dt")

        if files.cargo.df is not None:
            # Индекс стоимости грузоперевозок
            # Данные предоставляются по средам, но чтобы их учитывать в тендерах
            # мы меняем их дату на следующий понедельник
            files.cargo.df = pd.concat(
                (
                    files.cargo.df,
                    all_df[["dt"]],
                )
            ).sort_values("dt")
            files.cargo.df["Индекс стоимости грузоперевозок"] = files.cargo.df[
                "Индекс стоимости грузоперевозок"
            ].shift(1)
            files.cargo.df = files.cargo.df[files.cargo.df.dt.isin(all_df.dt)]
            all_df = all_df.merge(files.cargo.df, how="left", on="dt")

        if files.metal_market.df is not None:
            # Данные у нас предоставляется в последний день каждого месяца
            # Меняем дату на следующие понедельник
            # Т.е. в понедельник определенного месяца будут одни и те же
            files.metal_market.df = pd.concat(
                (
                    files.metal_market.df,
                    all_df[~all_df.dt.isin(files.metal_market.df.dt)][["dt"]],
                )
            ).sort_values("dt")
            files.metal_market.df = files.metal_market.df.fillna(method="ffill")
            files.metal_market.df = files.metal_market.df[
                files.metal_market.df.dt.isin(all_df.dt)
            ]
            files.metal_market.df.columns = [
                f"металл_{x}" if files.metal_market.df.columns[x] != "dt" else "dt"
                for x in range(len(files.metal_market.df.columns))
            ]
            all_df = all_df.merge(files.metal_market.df, how="left", on="dt")

        if files.construction.df is not None:
            # То же самое, что и с показателями ранка металла
            files.construction.df = pd.concat(
                (
                    files.construction.df,
                    all_df[~all_df.dt.isin(files.construction.df.dt)][["dt"]],
                )
            ).sort_values("dt")
            files.construction.df = files.construction.df.fillna(method="ffill")
            files.construction.df = files.construction.df[
                files.construction.df.dt.isin(all_df.dt)
            ].dropna()
            files.construction.df.columns = [
                f"материалы_{x}" if files.construction.df.columns[x] != "dt" else "dt"
                for x in range(len(files.construction.df.columns))
            ]
            all_df = all_df.merge(files.construction.df, how="left", on="dt")

        if files.fuel.df is not None:
            # То же самое
            toplivo = files.fuel.df
            toplivo = pd.concat(
                (toplivo, all_df[~all_df.dt.isin(toplivo.dt)][["dt"]])
            ).sort_values("dt")
            toplivo = toplivo.fillna(method="ffill")
            toplivo = toplivo[toplivo.dt.isin(all_df.dt)].dropna()
            toplivo.columns = [
                f"топливо_{x}" if toplivo.columns[x] != "dt" else "dt"
                for x in range(len(toplivo.columns))
            ]
            all_df = all_df.merge(toplivo, how="left", on="dt")

        if files.lme_index.df is not None:
            # И опять:)
            lme = files.lme_index.df.rename(
                {"дата": "dt", "цена": "индекс lme"}, axis=1
            ).dropna()
            lme = pd.concat((lme, all_df[~all_df.dt.isin(lme.dt)][["dt"]])).sort_values(
                "dt"
            )
            lme = lme.fillna(method="ffill")
            lme = lme[lme.dt.isin(all_df.dt)].dropna()
            all_df = all_df.merge(lme, how="left", on="dt")

        if files.chmf.df is not None:
            # Берем цену акции, считаем как изменилась цена за предыдущую неделю
            chmf_stock = files.chmf.df[["Date", "Change %"]].rename(
                {"Date": "dt", "Change %": "chmf_change"}, axis=1
            )
            chmf_stock["chmf_change"] = (
                chmf_stock["chmf_change"].str.replace("%", "").astype(np.float32) / 100
                + 1
            )
            # Собираем информацию по изменению цены за предыдущие 5 дней
            chmf_stock["chmf_change"] = (
                chmf_stock.chmf_change
                * chmf_stock.chmf_change.shift(-1)
                * chmf_stock.chmf_change.shift(-2)
                * chmf_stock.chmf_change.shift(-3)
                * chmf_stock.chmf_change.shift(-4)
            )
            # Берем данные за пятницу и меняем дату на следующий понедельник,
            # когда будет проводиться тендер
            chmf_stock = chmf_stock[chmf_stock.dt.dt.day_of_week == 4].dropna()
            chmf_stock = pd.concat(
                (chmf_stock, all_df[~all_df.dt.isin(chmf_stock.dt)][["dt"]])
            ).sort_values("dt")
            chmf_stock = chmf_stock.fillna(method="ffill")
            chmf_stock = chmf_stock[chmf_stock.dt.isin(all_df.dt)].dropna()
            all_df = all_df.merge(chmf_stock, how="left", on="dt")

        if files.magn.df is not None:
            # То же самое
            magn_stock = files.magn.df[["Дата", "Изм. %"]].rename(
                {"Дата": "dt", "Изм. %": "magn_change"}, axis=1
            )
            magn_stock["magn_change"] = (
                magn_stock["magn_change"]
                .str.replace("%", "")
                .str.replace(",", ".")
                .astype(np.float32)
                / 100
                + 1
            )
            magn_stock["magn_change"] = (
                magn_stock.magn_change
                * magn_stock.magn_change.shift(-1)
                * magn_stock.magn_change.shift(-2)
                * magn_stock.magn_change.shift(-3)
                * magn_stock.magn_change.shift(-4)
            )
            magn_stock = magn_stock[magn_stock.dt.dt.day_of_week == 4].dropna()
            magn_stock = pd.concat(
                (magn_stock, all_df[~all_df.dt.isin(magn_stock.dt)][["dt"]])
            ).sort_values("dt")
            magn_stock = magn_stock.fillna(method="ffill")
            magn_stock = magn_stock[magn_stock.dt.isin(all_df.dt)].dropna()
            all_df = all_df.merge(magn_stock, how="left", on="dt")

        if files.nlmk.df is not None:
            # То же самое:)
            nlmk_stock = files.nlmk.df[["Date", "Change %"]].rename(
                {"Date": "dt", "Change %": "nlmk_change"}, axis=1
            )
            nlmk_stock["nlmk_change"] = (
                nlmk_stock["nlmk_change"]
                .str.replace("%", "")
                .str.replace(",", ".")
                .astype(np.float32)
                / 100
                + 1
            )
            nlmk_stock["nlmk_change"] = (
                nlmk_stock.nlmk_change
                * nlmk_stock.nlmk_change.shift(-1)
                * nlmk_stock.nlmk_change.shift(-2)
                * nlmk_stock.nlmk_change.shift(-3)
                * nlmk_stock.nlmk_change.shift(-4)
            )
            nlmk_stock = nlmk_stock[nlmk_stock.dt.dt.day_of_week == 4].dropna()
            nlmk_stock = pd.concat(
                (nlmk_stock, all_df[~all_df.dt.isin(nlmk_stock.dt)][["dt"]])
            ).sort_values("dt")
            nlmk_stock = nlmk_stock.fillna(method="ffill")
            nlmk_stock = nlmk_stock[nlmk_stock.dt.isin(all_df.dt)].dropna()
            all_df = all_df.merge(nlmk_stock, how="left", on="dt")

            all_df["Изменение цены на арматуру"] = all_df["Цена на арматуру"] / all_df[
                "Цена на арматуру"
            ].shift(1)
            all_df["month"] = all_df.dt.dt.month
            all_df = all_df.dropna(axis=1, how="all")

        return all_df

    def _make_features(self, df: pd.DataFrame):
        """Создание фичей по имеющимся у нас данным

        Args:
            df: Данные, которые у нас имеются для рекомендации к тендеру

        Returns:
            list: Список из фичей, которые достали из данных
        """
        cur_values = []

        # Как в последние 18 недель изменялась цена на арматуру
        cur_change = df["Изменение цены на арматуру"][-18:].tolist()
        cur_values.append(cur_change)

        # Во сколько раз изменилась цена на арматуру за последние 18 недель
        a = 1
        for x in cur_change:
            a *= x
        cur_values.append([a])

        # Средняя цена на арматуру в последние 10 недель
        cur_values.append([np.mean(df["Цена на арматуру"][-10:])])
        cur_values.append([df["Изменение в месяце"].tolist()[-1]])

        # Для каждой колонке из таблицы "Цены на сырье" вычисляем во сколько раз
        # изменилась цена за последние 1,2 месяца и за последнюю неделю
        cur_values.append(
            [
                df[x].tolist()[-1] / df[x].tolist()[-10]
                if df[x].tolist()[-10] != 0
                else np.nan
                for x in df.columns
                if "сырье" in x
            ]
        )
        cur_values.append(
            [
                df[x].tolist()[-1] / df[x].tolist()[-5]
                if df[x].tolist()[-5] != 0
                else np.nan
                for x in df.columns
                if "сырье" in x
            ]
        )
        cur_values.append(
            [
                df[x].tolist()[-1] / df[x].tolist()[-2]
                if df[x].tolist()[-2] != 0
                else np.nan
                for x in df.columns
                if "сырье" in x
            ]
        )

        # Вычисляем на сколько и во сколько раз изменился индекс стоимости
        # грузоперевозок за последнюю неделю и 1,2 месяца

        cur_values.append(
            [
                df["Индекс стоимости грузоперевозок"].tolist()[-1]
                - df["Индекс стоимости грузоперевозок"].tolist()[-2],
                df["Индекс стоимости грузоперевозок"].tolist()[-1]
                - df["Индекс стоимости грузоперевозок"].tolist()[-5],
                df["Индекс стоимости грузоперевозок"].tolist()[-1]
                - df["Индекс стоимости грузоперевозок"].tolist()[-10],
                df["Индекс стоимости грузоперевозок"].tolist()[-1]
                - df["Индекс стоимости грузоперевозок"].tolist()[-20],
                df["Индекс стоимости грузоперевозок"].tolist()[-1]
                / df["Индекс стоимости грузоперевозок"].tolist()[-2],
                df["Индекс стоимости грузоперевозок"].tolist()[-1]
                / df["Индекс стоимости грузоперевозок"].tolist()[-5],
                df["Индекс стоимости грузоперевозок"].tolist()[-1]
                / df["Индекс стоимости грузоперевозок"].tolist()[-10],
                df["Индекс стоимости грузоперевозок"].tolist()[-1]
                / df["Индекс стоимости грузоперевозок"].tolist()[-20],
            ]
        )

        # Для каждой колонке из таблицы "Показатели рынка металла" считаем во
        # сколько раз  изменилась цена за последние 1,2,4 месяца и берем среднее
        # значение
        cur_values.append(
            [
                # За последний месяц
                np.nanmean(
                    [
                        df[x].tolist()[-1] / df[x].tolist()[-5]
                        if df[x].tolist()[-5] != 0
                        else np.nan
                        for x in df.columns
                        if "металл" in x
                    ]
                ),
                # За последние 2 месяца
                np.nanmean(
                    [
                        df[x].tolist()[-1] / df[x].tolist()[-10]
                        if df[x].tolist()[-10] != 0
                        else np.nan
                        for x in df.columns
                        if "металл" in x
                    ]
                ),
                # За последние 4 месяца
                np.nanmean(
                    [
                        df[x].tolist()[-1] / df[x].tolist()[-20]
                        if df[x].tolist()[-20] != 0
                        else np.nan
                        for x in df.columns
                        if "металл" in x
                    ]
                ),
            ]
        )

        # Для каждой колонке из таблицы о стройматериалах высчитываем во склько раз
        # изменилась цена за последние 1,2,4 месяца
        cur_values.append(
            [
                np.nanmean(
                    [
                        df[x].tolist()[-1] / df[x].tolist()[-10]
                        if df[x].tolist()[-10] != 0
                        else np.nan
                        for x in df.columns
                        if "материалы" in x
                    ]
                ),
                np.nanmean(
                    [
                        df[x].tolist()[-1] / df[x].tolist()[-5]
                        if df[x].tolist()[-5] != 0
                        else np.nan
                        for x in df.columns
                        if "материалы" in x
                    ]
                ),
                np.nanmean(
                    [
                        df[x].tolist()[-1] / df[x].tolist()[-20]
                        if df[x].tolist()[-20] != 0
                        else np.nan
                        for x in df.columns
                        if "материалы" in x
                    ]
                ),
            ]
        )
        # То же самое, только для топлива
        cur_values.append(
            [
                df[x].tolist()[-1] - df[x].tolist()[-10]
                if df[x].tolist()[-10] != 0
                else np.nan
                for x in df.columns
                if "топливо" in x
            ]
        )
        cur_values.append(
            [
                df[x].tolist()[-1] - df[x].tolist()[-20]
                if df[x].tolist()[-20] != 0
                else np.nan
                for x in df.columns
                if "топливо" in x
            ]
        )

        # На сколько пунктов изменился индекс lme за последние 1,2,4 месяца
        cur_values.append(
            [
                df["индекс lme"].tolist()[-1] - df["индекс lme"].tolist()[-5],
                df["индекс lme"].tolist()[-1] - df["индекс lme"].tolist()[-10],
                df["индекс lme"].tolist()[-1] - df["индекс lme"].tolist()[-20],
            ]
        )

        # Как изменилась цена акций за последние 5 недель
        cur_values.append(
            [
                np.nanprod(df["chmf_change"].tolist()[-5:]),
                np.nanprod(df["magn_change"].tolist()[-5:]),
                np.nanprod(df["nlmk_change"].tolist()[-5:]),
            ]
        )

        return np.concatenate(cur_values)
