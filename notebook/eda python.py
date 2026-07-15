"""
Exploratory Data Analysis — Online Course dataset.

Converted from EDA.ipynb into a standalone script. Run directly to reproduce
all the notebook's checks and plots:

    python eda.py --data dataset/online_course.xlsx
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    print(df.head())
    print(df.tail())
    return df


# --------------------------------------------------------------------------- #
# Basic info / data quality checks
# --------------------------------------------------------------------------- #
def basic_info(df: pd.DataFrame) -> None:
    # checking DataFrame informations
    df.info()
    print(df.describe())


def check_nulls_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    # Finding Null values
    print(df.isnull().sum())

    # Drop Null Values if Exists
    df = df.dropna().copy()

    # Count Duplicates
    print(f"Duplicate rows: {df.duplicated().sum()}")

    # here the time_spent for course is greater than course_duration
    print(df[df["course_duration_hours"] < df["time_spent_hours"]].head(5))
    return df


# --------------------------------------------------------------------------- #
# Categorical (object) columns
# --------------------------------------------------------------------------- #
def categorical_analysis(df: pd.DataFrame) -> None:
    # Seperate Object Columns for EDA
    obj_cols = df.select_dtypes(include="object")
    print(obj_cols.head(1))

    # Number of Unique values in Object columns
    for col in obj_cols:
        print(f"\n{col}")
        print(df[col].nunique())

    # Ploting the Object Columns to finding the majority between the features
    for col in obj_cols:
        plt.figure(figsize=(10, 7))

        ax = sns.countplot(
            data=df,
            x=col,
            order=df[col].value_counts().index,
        )

        for container in ax.containers:
            ax.bar_label(container)

        plt.xticks(rotation=90)
        plt.title(col)
        plt.tight_layout()
        plt.show()

    # how many unique users and uniques course
    users_count = df["user_id"].nunique()
    course_count = df["course_name"].nunique()
    print(f"how many Unique users: {users_count}")
    print(f"how many Unique courses: {course_count}")

    # the user with more previous course taken
    print(df.loc[df["previous_courses_taken"].idxmax()])


# --------------------------------------------------------------------------- #
# Numerical columns
# --------------------------------------------------------------------------- #
def numerical_analysis(df: pd.DataFrame) -> pd.DataFrame:
    # Seperate numerical columns
    num_cols = df.select_dtypes(include=["int", "float64"])
    num_cols = num_cols.drop(columns=["user_id", "course_id"], axis=1)
    df.info()

    # checking that time_spent_hours by most of users
    df["time_spent_hours"].hist()
    plt.show()

    # the most of the users given feedback above 4 point out of 10
    df["feedback_score"].hist()
    plt.show()

    # ploting the rating feature
    plt.figure(figsize=(13, 9))
    sns.countplot(data=df, x="rating")
    plt.show()

    print(num_cols.head())
    return num_cols


def outlier_analysis(df: pd.DataFrame, num_cols: pd.DataFrame) -> None:
    # Finding the Outliers
    for col in num_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, y=col)
        plt.title(f"Box Plot of {col}")
        plt.show()


def correlation_analysis(df: pd.DataFrame, num_cols: pd.DataFrame) -> None:
    # finding the Correlation between the numerical columns
    cor = df[num_cols.columns].corr()
    plt.figure(figsize=(8, 11))
    sns.heatmap(cor, annot=True, cmap="coolwarm", fmt=".4f")
    plt.title("correlation HeateMap")
    plt.show()

    corrmatrix = cor.unstack()
    corrmatrix = corrmatrix[corrmatrix != 1]
    corrmatrix = np.round(corrmatrix.sort_values(ascending=True), 2)
    print(corrmatrix.head())

    # There is no Coorelation between the features
    print("There is no correlation between the features")


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Run EDA on the online course dataset.")
    parser.add_argument("--data", default="dataset/online_course.xlsx",
                         help="Path to the course dataset (.xlsx)")
    args = parser.parse_args()

    df = load_data(args.data)
    basic_info(df)
    df = check_nulls_and_duplicates(df)
    categorical_analysis(df)
    num_cols = numerical_analysis(df)
    outlier_analysis(df, num_cols)
    correlation_analysis(df, num_cols)


if __name__ == "__main__":
    main()
