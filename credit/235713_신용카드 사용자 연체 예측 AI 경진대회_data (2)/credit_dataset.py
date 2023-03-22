import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class CreditDataSet(Dataset):
    def __init__(self, csv_path, means=None, stds=None, categories=None, data_type='train') -> None:
        super().__init__()

        self.drop_column = ['index', 'family_size', 'FLAG_MOBIL']
        self.category_column = ['gender', 'car', 'reality', 'income_type', 'edu_type', 'family_type', 'house_type', 'occyp_type']
        self.embedding_column = self.category_column + ['child_num', 'work_phone', 'phone', 'email']
        self.scale_column = ['income_total', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'begin_month']

        df = pd.read_csv(csv_path)
        df.fillna('NoJob', inplace=True)  # 결측치 보충
        df.drop(self.drop_column, axis=1, inplace=True)  # 필요없는 컬럼 제거

        df = self._manual_preprocess(df)  # 전처리
        df, self.categories = self._category_to_num(df, self.category_column, categories)  # 문자열 -> 정수
        df, self.means, self.stds = self._standard_scale(df, self.scale_column, means, stds)  # 표준화

        one_hot_df = pd.get_dummies(df[self.embedding_column])  # 원 핫
        if data_type == 'train':
            df = pd.concat([one_hot_df, df[self.scale_column + ['credit']]], axis=1)
        else:
            df = pd.concat([one_hot_df, df[self.scale_column]], axis=1)
        self.df = df

        # self.embedding_column_idx = [list(df.columns).index(col) for col in self.embedding_column]
        # self.embedding_column_dimension = [df[col].nunique() for col in self.embedding_column]

        self.data_type = data_type
        self.datas = df.values.astype(np.float32)
        if data_type == 'train':
            self.x = self.datas[:, :-1]
            self.y = self.datas[:, -1]
            self.x = torch.from_numpy(self.x)
            self.y = torch.from_numpy(self.y)
        else:
            self.x = self.datas
            self.x = torch.from_numpy(self.x)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        if self.data_type == 'train':
            return self.x[index], self.y[index]
        else:
            return self.x[index]
        
    def _manual_preprocess(self, df):
        df.loc[df['child_num'] >= 5, 'child_num'] = 5  # 자식수 5명 이상이면 전부 5로 퉁침
        df.loc[df['DAYS_EMPLOYED'] == 365243, 'DAYS_EMPLOYED'] = 0  # 365243 이건 왜 나온지 모르겠음
        return df

    def _category_to_num(self, df, category_column, trained_categories=None):
        if trained_categories is None:
            categories = []
        else:
            categories = trained_categories

        for i, column in enumerate(category_column):
            if trained_categories is None:
                catergory = df[column].unique()
                categories.append(catergory)
            else:
                catergory = categories[i]

            for n, x in enumerate(catergory):
                df.loc[df[column] == x, column] = n

        return df, categories

    def _standard_scale(self, df, column, means=None, stds=None):
        standard_df = df[column]
        if means is None:
            means = standard_df.mean()
            stds = standard_df.std()
        
        standard_df = (standard_df - means) / stds
        df[column] = standard_df
        return df, means, stds


# print(df.nunique())
# print(df.info())
# print(df.describe())

if __name__ == '__main__':
    datasets = CreditDataSet('train.csv')
    x, y = datasets[2]
    # print(datasets.df)
    print(x, y)