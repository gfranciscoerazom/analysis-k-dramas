import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import upsetplot

@pd.api.extensions.register_dataframe_accessor("missing_data")
class MissingData:
    def __init__(self, df: pd.DataFrame):
        self.df: pd.DataFrame = df
        self.is_na_df: pd.DataFrame = df.isna()

        self.count_na_per_column: pd.Series = self.is_na_df.sum(axis=0)
        self.count_na_per_row: pd.Series = self.is_na_df.sum(axis=1)

        self.count_na_per_column_percentage: pd.Series = self.count_na_per_column / self.df.shape[0] * 100
        self.count_na_per_row_percentage: pd.Series = self.count_na_per_row / self.df.shape[1] * 100

        self.total_count_na: int = self.count_na_per_column.sum()
        self.total_count_not_na: int = self.df.size - self.total_count_na

        self.total_count_na_percentage: float = self.total_count_na / self.df.size * 100
        self.total_count_not_na_percentage: float = self.total_count_not_na / self.df.size * 100

        self.columns_with_na = self.count_na_per_column[self.count_na_per_column > 0].index
        self.columns_without_na = self.count_na_per_column[self.count_na_per_column == 0].index


    def na_count_and_percentage_per(self, axis: str = "column") -> pd.DataFrame:
        if axis == "column":
            return pd.DataFrame({
                "count": self.count_na_per_column,
                "percentage": self.count_na_per_column_percentage
            })
        elif axis == "row":
            return pd.DataFrame({
                "count": self.count_na_per_row,
                "percentage": self.count_na_per_row_percentage
            })
        else:
            raise ValueError("axis must be either 'column' or 'row'")


    def coincidence_count_na_per(self, axis: str = "column") -> pd.DataFrame:
        if axis == "column":
            return self.count_na_per_column.value_counts().sort_index()
        elif axis == "row":
            return self.count_na_per_row.value_counts().sort_index()
        else:
            raise ValueError("axis must be either 'column' or 'row'")


    def coincidence_and_percentage_count_na_per(self, axis: str = "column") -> pd.DataFrame:
        if axis == "column":
            return pd.DataFrame({
                "count": self.coincidence_count_na_per(axis="column"),
                "percentage": self.coincidence_count_na_per(axis="column") / self.df.shape[1] * 100
            })
        elif axis == "row":
            return pd.DataFrame({
                "count": self.coincidence_count_na_per(axis="row"),
                "percentage": self.coincidence_count_na_per(axis="row") / self.df.shape[0] * 100
            })
        else:
            raise ValueError("axis must be either 'column' or 'row'")


    def na_count_by_intervals(self, interval: int, column: str) -> pd.DataFrame:
        intervals_df: pd.DataFrame = self.df.assign(groupby_number=lambda df: np.repeat(np.arange(self.df.shape[0]), interval)[:self.df.shape[0]]).groupby("groupby_number").aggregate(size_of_interval=(column, "size"), count_of_na=(column, lambda inter: inter.isna().sum()))
        intervals_df = intervals_df.assign(
            count_of_not_na=lambda df: df["size_of_interval"] - df["count_of_na"],
            percentage_of_na=lambda df: df["count_of_na"] / df["size_of_interval"] * 100,
            percentage_of_not_na=lambda df: df["count_of_not_na"] / df["size_of_interval"] * 100
        ).drop(columns=["size_of_interval"])
        return intervals_df


    def na_count_by_bins(self, bins: int, column: str) -> pd.DataFrame:
        intervals_df: pd.DataFrame = self.df.assign(groupby_number=lambda df: pd.cut(np.arange(self.df.shape[0]), bins=bins, labels=False)[:self.df.shape[0]]).groupby("groupby_number").aggregate(size_of_interval=(column, "size"), count_of_na=(column, lambda inter: inter.isna().sum()))
        intervals_df = intervals_df.assign(
            count_of_not_na=lambda df: df["size_of_interval"] - df["count_of_na"],
            percentage_of_na=lambda df: df["count_of_na"] / df["size_of_interval"] * 100,
            percentage_of_not_na=lambda df: df["count_of_not_na"] / df["size_of_interval"] * 100
        ).drop(columns=["size_of_interval"])
        return intervals_df


    def size_of_sections_of_na_and_not_na(self, column: str) -> pd.DataFrame:
        rle_list = [(len(list(group)), key) for key, group in itertools.groupby(self.df[column].isna())]

        return(
            pd.DataFrame(
                rle_list,
                columns=['num_in_section', 'value']
            )
            .replace({
                False: 'not_na',
                True: 'na'
            })
        )


    def plot_of_na_count_per_column(self) -> None:
            df_to_plot = self.count_na_per_column.sort_values()
            plt.hlines(
                y=df_to_plot.index,
                xmin=0,
                xmax=df_to_plot.values,
                color="black"
            )
            plt.plot(
                df_to_plot.values,
                df_to_plot.index,
                "o",
                color="black"
            )
            plt.xlabel("Count of na")
            plt.ylabel("Column")
            plt.title("Count na per column")
            plt.grid(axis="x")
            plt.show()


    def histplot_of_na_count_per_row(self) -> None:
        df_to_plot = self.count_na_per_row

        sns.histplot(
            df_to_plot,
            binwidth=1,
            color="black"
        )

        plt.xlabel("Count of na")
        plt.ylabel("Count of rows")
        plt.title("Count na per row")
        plt.grid(axis="y")
        plt.show()


    def percentage_of_not_na_vs_percentage_of_na_plot(self, interval: int, column: str, rot: int=0, figsize=None) -> None:
        df_to_plot = self.na_count_by_intervals(interval, column)

        df_to_plot.plot.barh(
            y=["percentage_of_not_na", "percentage_of_na"],
            stacked=True,
            width=1,
            color=["black", "gray"],
            rot=rot,
            figsize=figsize
        )

        plt.xlabel("Interval")
        plt.ylabel("Count of na")
        plt.title("Count na by intervals")
        plt.grid(axis="x")
        plt.margins(0)
        plt.tight_layout(pad=0)
        plt.show()


    def upsetplot(self, variable: list[str] = None, **kwargs):
        if variable is None:
            variable = self.columns_with_na.to_list()

        df_to_plot = self.is_na_df.value_counts(variable)

        return upsetplot.plot(
            df_to_plot,
            **kwargs
        )
