from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


# =======================
# Função de recomendação
# =======================
def build_recommendations(movies, user_history, liked_genres, alpha=0.2, beta=0.4, gamma=0.4, top_n=20):
    df_movies = pd.DataFrame(movies)

    if 'overview' not in df_movies.columns:
        df_movies['overview'] = ""
    if 'genres' not in df_movies.columns:
        df_movies['genres'] = [[] for _ in range(len(df_movies))]
    if 'vote_average' not in df_movies.columns:
        df_movies['vote_average'] = 5.0
    if 'vote_count' not in df_movies.columns:
        df_movies['vote_count'] = 0

    # -----------------------
    # 0. Filtrar filmes com vote_count >= 20
    # -----------------------
    df_movies = df_movies[df_movies['vote_count'] >= 20]

    if df_movies.empty:
        return []

    # -----------------------
    # 1. Representação de gêneros
    # -----------------------
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(df_movies['genres'])
    genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_)

    # -----------------------
    # 2. Representação de descrições
    # -----------------------
    tfidf = TfidfVectorizer(stop_words="english")
    desc_matrix = tfidf.fit_transform(df_movies['overview'])

    # -----------------------
    # 3. Perfil do usuário
    # -----------------------
    user_genre_pref = np.zeros(len(mlb.classes_))
    user_desc_pref = np.zeros((1, desc_matrix.shape[1]))
    seen_ids = set()

    for hist in user_history:
        seen_ids.add(int(hist.get("movieId")))
        liked = hist.get("liked", False)
        genres = hist.get("movieGenre", [])
        desc = hist.get("movieDescription", "")

        if liked:
            # gostou → positivo
            genre_weight = 5    # positivo mediano
            desc_weight = 10    # positivo alto
        else:
            # não gostou → negativo
            genre_weight = -4   # negativo mediano
            desc_weight = -15   # negativo muito alto

        genre_vec = mlb.transform([genres])[0]
        user_genre_pref += genre_vec * genre_weight

        desc_vec = tfidf.transform([desc])
        user_desc_pref += desc_vec.toarray() * desc_weight

    # gêneros favoritos declarados → maior peso positivo
    if liked_genres:
        liked_genre_vec = mlb.transform([liked_genres])[0]
        user_genre_pref += liked_genre_vec * 20  # peso altíssimo

    # normalização
    if np.linalg.norm(user_genre_pref) > 0:
        user_genre_pref = user_genre_pref / np.linalg.norm(user_genre_pref)
    if np.linalg.norm(user_desc_pref) > 0:
        user_desc_pref = user_desc_pref / np.linalg.norm(user_desc_pref)

    # -----------------------
    # 4. Similaridade
    # -----------------------
    sim_genre = cosine_similarity([user_genre_pref], genre_df)[0]
    sim_desc = cosine_similarity(user_desc_pref, desc_matrix)[0]

    # -----------------------
    # 5. Score híbrido
    # -----------------------
    df_movies["genre_score"] = sim_genre
    df_movies["desc_score"] = sim_desc
    df_movies["final_score"] = (
        alpha * df_movies["vote_average"] +
        beta * df_movies["genre_score"] +
        gamma * df_movies["desc_score"]
    )

    # -----------------------
    # 6. Ordenar e remover filmes já assistidos
    # -----------------------
    candidates = df_movies[~df_movies['id'].isin(seen_ids)].sort_values("final_score", ascending=False)

    # -----------------------
    # 7. Diversidade + seleção principal
    # -----------------------
    final_list = []
    used_genres = set()
    for _, row in candidates.iterrows():
        if len(final_list) >= top_n:
            break
        genres = set(row['genres'])
        if not genres.issubset(used_genres) or len(final_list) < top_n // 2:
            final_list.append(row.to_dict())
            used_genres.update(genres)

    # -----------------------
    # 8. Completar se faltar
    # -----------------------
    if len(final_list) < top_n:
        remaining = candidates[~candidates['id'].isin([m['id'] for m in final_list])].head(top_n - len(final_list))
        final_list.extend(list(remaining.to_dict('records')))

    # -----------------------
    # 9. Serendipidade + exploração
    # -----------------------
    if len(final_list) < top_n and not candidates.empty:
        mid_idx = len(candidates) // 2
        ser_item = candidates.iloc[mid_idx]
        if ser_item['id'] not in [m['id'] for m in final_list]:
            final_list.append(ser_item.to_dict())

    if len(final_list) < top_n and not candidates.empty:
        popular_item = candidates.sort_values("popularity", ascending=False).iloc[0]
        if popular_item['id'] not in [m['id'] for m in final_list]:
            final_list.append(popular_item.to_dict())

    return final_list[:top_n]




# =======================
# Endpoint Flask
# =======================
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        payload = request.json
        movies = payload.get("movies", [])
        user_history = payload.get("userHistory", [])
        liked_genres = payload.get("genres", [])

        if not movies:
            return jsonify({"error": "Nenhum filme fornecido"}), 400

        recommendations = build_recommendations(movies, user_history, liked_genres)
        return jsonify({"recommendations": recommendations})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

