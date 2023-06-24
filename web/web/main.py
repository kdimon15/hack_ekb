import datetime

import altair as alt
import pandas as pd
import streamlit as st
from model import DateIsNotPresentException, Model
from models.file_entry import FileEntry
from models.files import Files


class App:
    def __init__(self, model: Model) -> None:
        self._files = Files()
        self._model = model
        self._init_sidebar_layout()
        self._init_main_layout()

    def _init_main_layout(self):
        st.write("# Хакатон")
        self.test_predict_tab, self.date_predict_tab, self.data_tab = st.tabs(
            [
                "Проверка на тестовом датасете",
                "Предсказать для даты",
                "Данные",
            ]
        )
        self._init_data_tab()
        self._init_test_predict_tab()
        self._init_date_predict_tab()

    def _init_data_tab(self):
        with self.data_tab:
            self._add_dataframe_display(self._files.cargo)
            self._add_dataframe_display(self._files.construction)
            self._add_dataframe_display(self._files.lme_index)
            self._add_dataframe_display(self._files.macro)
            self._add_dataframe_display(self._files.metal_market)
            self._add_dataframe_display(self._files.fuel)
            self._add_dataframe_display(self._files.raw_prices)
            self._add_dataframe_display(self._files.chmf)
            self._add_dataframe_display(self._files.magn)
            self._add_dataframe_display(self._files.nlmk)
            self._add_dataframe_display(self._files.test)
            self._add_dataframe_display(self._files.train)

    def _init_test_predict_tab(self):
        with self.test_predict_tab:
            if self._files.is_uploaded:
                if st.button("Запустить", key="test_predict_button"):
                    score, graph = self._model.predict(
                        files=self._files,
                    )

                    st.metric(label="Метрика", value=score)
                    self._show_plot(graph)
            else:
                st.write("Загружены не все файлы")

    def _init_date_predict_tab(self):
        with self.date_predict_tab:
            date_to = st.date_input("Дата закупки, для которой делается прогноз")
            if date_to is None:
                return

            if self._files.is_uploaded:
                if st.button("Запустить", key="date_predict_button"):
                    try:
                        prediction = self._model.make_prediction_for_date(
                            files=self._files,
                            date_to=date_to,
                        )
                        st.metric("Рекомендация", prediction)
                    except DateIsNotPresentException:
                        st.write("Неверная дата")
            else:
                st.write("Загружены не все файлы")

    def _show_plot(self, graph: pd.DataFrame):
        hover = alt.selection_single(
            fields=["Дата"],
            nearest=True,
            on="mouseover",
            empty="none",
        )

        chart = (
            alt.Chart(graph)
            .mark_line()
            .encode(
                alt.Y("Стоимость").scale(zero=False),
                x="Дата",
                color="Тип",
            )
            .interactive()
        )

        tooltips = (
            alt.Chart(graph)
            .mark_rule()
            .encode(
                x="Дата",
                y="Стоимость",
                opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
                tooltip=[
                    alt.Tooltip("Дата", title="Дата"),
                    alt.Tooltip("Стоимость", title="Стоимость"),
                    alt.Tooltip("Тип", title="Тип"),
                ],
            )
            .add_selection(hover)
        )

        st.altair_chart(
            (chart + tooltips).interactive(),
            use_container_width=True,
        )

    def _init_sidebar_layout(self):
        with st.sidebar:
            self._add_file_uploader(self._files.cargo)
            self._add_file_uploader(self._files.construction)
            self._add_file_uploader(self._files.lme_index)
            self._add_file_uploader(self._files.macro)
            self._add_file_uploader(self._files.metal_market)
            self._add_file_uploader(self._files.fuel)
            self._add_file_uploader(self._files.raw_prices)
            self._add_file_uploader(self._files.chmf)
            self._add_file_uploader(self._files.magn)
            self._add_file_uploader(self._files.nlmk)
            self._add_file_uploader(self._files.test)
            self._add_file_uploader(self._files.train)

    def _add_file_uploader(
        self,
        file_entry: FileEntry,
    ):
        uploaded_file = st.file_uploader(file_entry.label, type=["xlsx", "csv"])
        if uploaded_file is not None:
            file_type = uploaded_file.name.split(".")[-1]
            if file_type == "xlsx":
                file_entry.df = pd.read_excel(
                    uploaded_file,
                    parse_dates=file_entry.parse_dates,
                )
            else:
                file_entry.df = pd.read_csv(
                    uploaded_file,
                    parse_dates=file_entry.parse_dates,
                )

    def _add_dataframe_display(self, file_entry: FileEntry):
        st.write(f"## {file_entry.label}")
        if file_entry.df is not None:
            st.dataframe(file_entry.df)
        else:
            st.write("Файл не загружен")


if __name__ == "__main__":
    model = Model("hack/models.pkl")
    App(model)
