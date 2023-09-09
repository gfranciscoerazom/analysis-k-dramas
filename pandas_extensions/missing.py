import itertools
import pandas as pd
import numpy as np

@pd.api.extensions.register_dataframe_accessor("missing_data")
class MissingData:
    def __init__(self, df: pd.DataFrame):
        self.df: pd.DataFrame = df
        self.num_of_rows: int = df.shape[0]
        self.num_of_columns: int = df.shape[1]

        self.is_na_df: pd.DataFrame = df.isna()

        self.count_na_per_column: pd.Series = self.is_na_df.sum(axis=0) #
        self.count_na_per_row: pd.Series = self.is_na_df.sum(axis=1) #

        self.count_na_per_column_percentage: pd.Series = self.count_na_per_column / self.df.shape[0] * 100 #
        self.count_na_per_row_percentage: pd.Series = self.count_na_per_row / self.df.shape[1] * 100 #

        self.total_count_na: int = self.count_na_per_column.sum() #
        self.total_count_not_na: int = self.df.size - self.total_count_na #

        self.total_count_na_percentage: float = self.total_count_na / self.df.size * 100 #
        self.total_count_not_na_percentage: float = self.total_count_not_na / self.df.size * 100 #


    def na_count_and_percentage_df(self, axis: str = "column") -> pd.DataFrame:
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
                "percentage": self.coincidence_count_na_per(axis="column") / self.num_of_columns * 100
            })
        elif axis == "row":
            return pd.DataFrame({
                "count": self.coincidence_count_na_per(axis="row"),
                "percentage": self.coincidence_count_na_per(axis="row") / self.num_of_rows * 100
            })
        else:
            raise ValueError("axis must be either 'column' or 'row'")


    def na_count_by_intervals(self, interval: int, column: str) -> pd.DataFrame:
        intervals_df: pd.DataFrame = self.df.assign(groupby_number=lambda df: np.repeat(np.arange(self.df.shape[0]), interval)[:self.df.shape[0]]).groupby("groupby_number").aggregate(size_of_interval=(column, "size"), count_na=(column, lambda inter: inter.isna().sum()))
        intervals_df = intervals_df.assign(
            count_of_not_na=lambda df: df["size_of_interval"] - df["count_na"],
            percentage_of_not_na=lambda df: df["count_of_not_na"] / df["size_of_interval"] * 100,
            percentage_of_na=lambda df: df["count_na"] / df["size_of_interval"] * 100
        ).drop(columns=["size_of_interval"])
        return intervals_df


    def na_count_by_bins(self, bins: int, column: str) -> pd.DataFrame:
        intervals_df: pd.DataFrame = self.df.assign(groupby_number=lambda df: pd.cut(np.arange(self.df.shape[0]), bins=bins, labels=False)[:self.df.shape[0]]).groupby("groupby_number").aggregate(size_of_interval=(column, "size"), count_na=(column, lambda inter: inter.isna().sum()))
        intervals_df = intervals_df.assign(
            count_of_not_na=lambda df: df["size_of_interval"] - df["count_na"],
            percentage_of_not_na=lambda df: df["count_of_not_na"] / df["size_of_interval"] * 100,
            percentage_of_na=lambda df: df["count_na"] / df["size_of_interval"] * 100
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