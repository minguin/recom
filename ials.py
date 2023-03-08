# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Revisiting the Performance of IALS on Item Recommendation Benchmarks."""

import concurrent.futures

from util.models import RecommendResult, Dataset
from src.base_recommender import BaseRecommender
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
import scipy.sparse as sp

np.random.seed(0)


class IALSDataset():
  """A class holding the train and test data."""

  def __init__(self, train_by_user, train_by_item, test, num_batches):
    """Creates a DataSet and batches it.

    Args:
      train_by_user: list of (user, items)
      train_by_item: list of (item, users)
      test: list of (user, history_items, target_items)
      num_batches: partitions each set using this many batches.
    """
    self.train_by_user = train_by_user
    self.train_by_item = train_by_item
    self.test = test
    self.num_users = len(train_by_user)
    self.num_items = len(train_by_item)
    self.user_batches = self._batch(train_by_user, num_batches)
    self.item_batches = self._batch(train_by_item, num_batches)
    self.test_batches = self._batch(test, num_batches)

  def _batch(self, xs, num_batches):
    batches = [[] for _ in range(num_batches)]
    for i, x in enumerate(xs):
      batches[i % num_batches].append(x)
    return batches


class IALS():
  """iALS solver."""

  def __init__(self, num_users, num_items, embedding_dim, reg,
               unobserved_weight, stddev):
    self.embedding_dim = embedding_dim
    self.reg = reg
    self.unobserved_weight = unobserved_weight
    self.user_embedding = np.random.normal(
        0, stddev, (num_users, embedding_dim))
    self.item_embedding = np.random.normal(
        0, stddev, (num_items, embedding_dim))
    self._update_user_gramian()
    self._update_item_gramian()

  def _update_user_gramian(self):
    self.user_gramian = np.matmul(self.user_embedding.T, self.user_embedding)

  def _update_item_gramian(self):
    self.item_gramian = np.matmul(self.item_embedding.T, self.item_embedding)

  def score(self, user_history):
    user_emb = project(
        user_history, self.item_embedding, self.item_gramian, self.reg,
        self.unobserved_weight)
    result = np.dot(user_emb, self.item_embedding.T)
    return result

  def train(self, ds):
    """Runs one iteration of the IALS algorithm.

    Args:
      ds: a DataSet object.
    """
    # Solve for the user embeddings
    self._solve(ds.user_batches, is_user=True)
    self._update_user_gramian()
    # Solve for the item embeddings
    self._solve(ds.item_batches, is_user=False)
    self._update_item_gramian()

  def _solve(self, batches, is_user):
    """Solves one side of the matrix."""
    if is_user:
      embedding = self.user_embedding
      args = (self.item_embedding, self.item_gramian, self.reg,
              self.unobserved_weight)
    else:
      embedding = self.item_embedding
      args = (self.user_embedding, self.user_gramian, self.reg,
              self.unobserved_weight)
    results = map_parallel(solve, batches, *args)
    for r in results:
      for user, emb in r.items():
        embedding[user, :] = emb

        
def map_parallel(fn, xs, *args):
  """Applies a function to a list, equivalent to [fn(x, *args) for x in xs]."""
  if len(xs) == 1:
    return [fn(xs[0], *args)]

  num_threads = len(xs)
  executor = concurrent.futures.ProcessPoolExecutor(num_threads)
  futures = [executor.submit(fn, x, *args) for x in xs]
  concurrent.futures.wait(futures)
  results = [future.result() for future in futures]
  return results


def solve(data_by_user, item_embedding, item_gramian, global_reg,
          unobserved_weight):
  user_embedding = {}
  for user, items in data_by_user:
    reg = global_reg *(len(items) + unobserved_weight * item_embedding.shape[0])
    user_embedding[user] = project(
        items, item_embedding, item_gramian, reg, unobserved_weight)
  return user_embedding


def project(user_history, item_embedding, item_gramian, reg, unobserved_weight):
  """Solves one iteration of the iALS algorithm."""
  if not user_history:
    raise ValueError("empty user history in projection")
  emb_dim = np.shape(item_embedding)[1]
  lhs = np.zeros([emb_dim, emb_dim])
  rhs = np.zeros([emb_dim])
  for item in user_history:
    item_emb = item_embedding[item]
    lhs += np.outer(item_emb, item_emb)
    rhs += item_emb

  lhs += unobserved_weight * item_gramian
  lhs = lhs + np.identity(emb_dim) * reg
  return np.linalg.solve(lhs, rhs)


class MFModel(IALS):

  def _predict_one(self, user, item):
    """Predicts the score of a user for an item."""
    return np.dot(self.user_embedding[user],
                  self.item_embedding[item])

  def predict(self, pairs, batch_size, verbose):
    """Computes predictions for a given set of user-item pairs.
    Args:
      pairs: A pair of lists (users, items) of the same length.
      batch_size: unused.
      verbose: unused.
    Returns:
      predictions: A list of the same length as users and items, such that
      predictions[i] is the models prediction for (users[i], items[i]).
    """
    del batch_size, verbose
    num_examples = len(pairs[0])
    assert num_examples == len(pairs[1])
    predictions = np.empty(num_examples)
    for i in range(num_examples):
      predictions[i] = self._predict_one(pairs[0][i], pairs[1][i])
    return predictions


class iALSRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, topk,**kwargs) -> RecommendResult:
        epochs = kwargs.get("epochs", 128)
        embedding_dim = kwargs.get("embedding_dim", 8)
        regularization = kwargs.get("regularization", 0.0)
        unobserved_weight = kwargs.get("unobserved_weight", 1.0)
        stddev = kwargs.get("stddev", 0.1)
        
        # 評価数の閾値
        minimum_num_rating = kwargs.get("minimum_num_rating", 0)

        filtered_movielens_train = dataset.train.groupby("movie_id").filter(
            lambda x: len(x["movie_id"]) >= minimum_num_rating
        )

        # 行列分解用に行列を作成する
        movielens_train_high_rating = filtered_movielens_train[dataset.train.rating >= 4]

        unique_user_ids = sorted(movielens_train_high_rating.user_id.unique())
        unique_movie_ids = sorted(movielens_train_high_rating.movie_id.unique())
        user_id2index = dict(zip(unique_user_ids, range(len(unique_user_ids))))
        movie_id2index = dict(zip(unique_movie_ids, range(len(unique_movie_ids))))

        movielens_matrix = sp.dok_matrix((len(unique_user_ids), len(unique_movie_ids)), dtype=np.float32)
        for i, row in movielens_train_high_rating.iterrows():
            user_index = user_id2index[row["user_id"]]
            movie_index = movie_id2index[row["movie_id"]]
            movielens_matrix[user_index, movie_index] = 1.0

        train_pos_pairs = np.column_stack(movielens_matrix.nonzero())
        train_by_user = defaultdict(list)
        train_by_item = defaultdict(list)
        for u, i in train_pos_pairs:
            train_by_user[u].append(i)
            train_by_item[i].append(u)

        train_by_user = list(train_by_user.items())
        train_by_item = list(train_by_item.items())

        train_ds = IALSDataset(train_by_user, train_by_item, [], 1)
        
        # モデルの初期化
        model = MFModel(len(unique_user_ids), len(unique_movie_ids),
                      embedding_dim, regularization,
                      unobserved_weight,
                      stddev / np.sqrt(embedding_dim))

        # 学習
        for epoch in range(epochs):
            # Training
            _ = model.train(train_ds)
        
        # 推薦
        test_pos_pairs = np.column_stack((movielens_matrix!=1).nonzero())
        df_pred = pd.DataFrame(test_pos_pairs,columns = ['user_index','movie_index'])
        df_pred['rating'] = model.predict(np.transpose(test_pos_pairs),0,0)
        
        index2user_id = dict(zip(range(len(unique_user_ids)),unique_user_ids))
        index2movie_id = dict(zip(range(len(unique_movie_ids)),unique_movie_ids))
        
        pred_user2items_id = dict(df_pred.sort_values('rating',ascending=False).groupby('user_index')['movie_index'].apply(list))
        
        pred_user2items = defaultdict(list)
        for user_index, movie_indexes in pred_user2items_id.items():
            for movie_index in movie_indexes:
                pred_user2items[index2user_id[user_index]].append(index2movie_id[movie_index])
        
        # IMFでは評価値の予測は難しいため、rmseの評価は行わない。（便宜上、テストデータの予測値をそのまま返す）
        return RecommendResult(dataset.test.rating, pred_user2items)


if __name__ == "__main__":
    iALSRecommender().run_sample()
