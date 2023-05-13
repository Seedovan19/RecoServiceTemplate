import pandas as pd

from annoy import AnnoyIndex


def get_recomendations_ANN(user_id, k_recs, n_nearest=20):
    f = 4  # Length of item vector that will be indexed
    fn = 'ANN_tree'
    ANN_tree = AnnoyIndex(f, 'angular')

    ANN_tree.load(fn, prefault=False)

    interactions = pd.read_pickle('interactions.pkl')
    items = pd.read_pickle('items.pkl')

    recs = pd.DataFrame()
    i = 0

    while len(recs) < k_recs:
        nearest_list = ANN_tree.get_nns_by_item(user_id, n_nearest)
        user_index = 0

        for id_nearest in nearest_list:
            # Чем ближе пользователь в ANN к исходному,
            # тем рекомендации приоритетнее
            user_index += 1

            df_nearest_user_interactions = interactions.loc[
                interactions['user_id'] == id_nearest
                ]['item_id']

            for item_id in df_nearest_user_interactions:
                current_user_rec = \
                    items.loc[items['item_id'] == item_id][
                        'title'].to_list()[
                        0]
                df = pd.DataFrame([
                    [item_id,
                     interactions.loc[
                         (interactions[
                              'item_id'] == item_id) & (
                             interactions['user_id'] == id_nearest)
                         ]['last_watch_dt'].to_list()[0],
                     interactions.loc[
                         (interactions[
                              'item_id'] == item_id) & (
                             interactions['user_id'] == id_nearest)
                         ]['weight'].to_list()[0],
                     user_index
                     ]],
                    columns=['item_id', 'last_watch_dt', 'weight',
                             'user_index']
                )
                recs = pd.concat([recs, df])

        # убираем из рекомендаций фильмы,
        # которые пользователи не смотрели достаточно времени
        recs = recs[recs.weight >= 2]

        # сортируем по времени
        recs = recs.sort_values(['last_watch_dt', 'user_index'],
                                ascending=[False, False])

        # повторяющиеся фильмы удаляем
        recs = recs.drop_duplicates(subset=['item_id'])

        recs = recs.head(k_recs)

        i += 1
        if i > 10:
            print('TIMEOUT: Не удалось найти полный список рекомендаций')

    recs = recs['item_id'].to_list()

    return recs
