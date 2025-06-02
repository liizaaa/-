import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import networkx as nx

# ----------------------------------------
# 1. Синтетический датасет IT-курсов
# ----------------------------------------

def generate_course_data(num_courses=50, seed=42):
    np.random.seed(seed)
    topics = [
        "Python", "Web Development", "Data Science", "Machine Learning",
        "Cybersecurity", "DevOps", "Databases", "Cloud Computing",
        "Algorithms", "Mobile Development"
    ]
    num_topics = len(topics)

    course_ids = [f"IT_C{i+1}" for i in range(num_courses)]
    course_names = [f"Course {i+1}: {topics[np.random.randint(0, num_topics)]}" for i in range(num_courses)]

    topic_vectors = []
    for _ in range(num_courses):
        selected = np.random.choice(range(num_topics), size=np.random.randint(1, 4), replace=False)
        vec = np.zeros(num_topics)
        vec[selected] = 1
        topic_vectors.append(vec)
    topic_matrix = np.array(topic_vectors)

    difficulty = np.random.randint(1, 6, size=num_courses)
    hours_per_week = np.random.randint(5, 21, size=num_courses)

    feature_columns = [f"Topic_{t}" for t in topics] + ["Difficulty", "HoursPerWeek"]
    features = np.hstack([topic_matrix, difficulty.reshape(-1, 1), hours_per_week.reshape(-1, 1)])
    course_features = pd.DataFrame(data=features, index=course_ids, columns=feature_columns)
    course_features.insert(0, "CourseName", course_names)
    course_features["DifficultyLabel"] = difficulty
    course_features["Topics"] = topic_matrix.tolist()
    return course_features

# ----------------------------------------
# 2. Генерация синтетических профилей и обучения ML-модели
# ----------------------------------------

def generate_synthetic_users(num_users=200, course_features=None, seed=24):
    np.random.seed(seed)
    user_profiles = []
    training_data = []
    topics = [col for col in course_features.columns if col.startswith("Topic_")]

    status_map = {"Школьник":0, "Студент":1, "Работаю":2, "Другое":3}
    level_map = {"Ничего не знаю":1, "Начинающий":2, "Средний":3, "Работаю в этой сфере":4, "Профи":5}

    for i in range(num_users):
        status = np.random.choice(list(status_map.keys()))
        level = np.random.choice(list(level_map.keys()))
        weekly_hours = np.random.randint(1, 41)
        goal = np.random.choice(["Для себя", "Учёба", "Работа"])
        pref_topics = list(np.random.choice([t.replace("Topic_", "") for t in topics], size=np.random.randint(1,4), replace=False))

        # формируем вектор пользователя
        user_vec = np.zeros(len(topics) + 2)
        for topic in pref_topics:
            idx = topics.index(f"Topic_{topic}")
            user_vec[idx] = 1
        user_vec[-2] = level_map[level]
        user_vec[-1] = weekly_hours

        # для каждого курса генерируем рейтинг с шумом
        for cid in course_features.index:
            course_vec = course_features.loc[cid, topics + ["Difficulty", "HoursPerWeek"]].values.astype(float)
            sim = cosine_similarity([user_vec], [course_vec])[0][0]
            rating = sim * 4 + 1 + np.random.normal(0, 0.5)
            rating = max(1, min(5, rating))
            train_features = np.concatenate([user_vec, course_vec])
            training_data.append(np.concatenate([train_features, [rating]]))

    columns = [f"U_{t}" for t in topics] + ["U_Level", "U_Hours"] + [f"C_{t}" for t in topics] + ["C_Difficulty", "C_Hours"] + ["Rating"]
    df_train = pd.DataFrame(training_data, columns=columns)
    return df_train

# ----------------------------------------
# 3. Модель обучения и граф зависимостей курсов
# ----------------------------------------

course_features = generate_course_data()
df_train = generate_synthetic_users(course_features=course_features)

X = df_train.drop(columns=["Rating"]).values
y = df_train["Rating"].values
ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
ml_model.fit(X, y)

# создаем граф зависимостей курсов
def build_course_graph(course_features):
    G = nx.DiGraph()
    for i, cid1 in enumerate(course_features.index):
        for j, cid2 in enumerate(course_features.index):
            if i != j:
                topics1 = np.array(course_features.loc[cid1, "Topics"])
                topics2 = np.array(course_features.loc[cid2, "Topics"])
                if np.dot(topics1, topics2) > 0 and course_features.loc[cid1, "DifficultyLabel"] < course_features.loc[cid2, "DifficultyLabel"]:
                    G.add_edge(cid1, cid2)
    return G

course_graph = build_course_graph(course_features)

# ----------------------------------------
# 4. Streamlit Web App
# ----------------------------------------

def get_difficulty_label(value):
    if value <= 2:
        return "Новичок"
    elif value == 3:
        return "Средний"
    else:
        return "Продвинутый"

def main():
    st.title("Персонализированные IT-учебные планы")
    st.write("Пожалуйста, введите информацию о себе, чтобы получить персонализированные рекомендации по курсам.")

    status = st.selectbox("Ваш статус:", ["Школьник", "Студент", "Работаю", "Другое"])
    level = st.selectbox("Уровень знаний:", ["Ничего не знаю", "Начинающий", "Средний", "Работаю в этой сфере", "Профи"])
    weekly_hours = st.slider("Сколько часов в неделю вы готовы уделять обучению?", 1, 40, 5)
    goal = st.selectbox("Какова ваша цель?", ["Для себя", "Подтянуть знания для учёбы", "Хочу устроиться на работу", "Другое"])

    preferences = st.multiselect("Интересующие темы:", [col.replace("Topic_", "") for col in course_features.columns if col.startswith("Topic_")])

    if st.button("Рекомендовать курсы (ML-модель)"):
        topics = [col for col in course_features.columns if col.startswith("Topic_")]
        user_vec = np.zeros(len(topics) + 2)
        level_map = {"Ничего не знаю":1, "Начинающий":2, "Средний":3, "Работаю в этой сфере":4, "Профи":5}
        for topic in preferences:
            idx = topics.index(f"Topic_{topic}")
            user_vec[idx] = 1
        user_vec[-2] = level_map[level]
        user_vec[-1] = weekly_hours

        scores = []
        for cid in course_features.index:
            course_vec = course_features.loc[cid, topics + ["Difficulty", "HoursPerWeek"]].values.astype(float)
            features = np.concatenate([user_vec, course_vec])
            pred_rating = ml_model.predict([features])[0]
            scores.append((cid, pred_rating))

        top = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
        st.subheader("Рекомендованные IT-курсы (по ML-модели):")
        for cid, score in top:
            cname = course_features.loc[cid, "CourseName"]
            difficulty_value = course_features.loc[cid, "DifficultyLabel"]
            difficulty_label = get_difficulty_label(difficulty_value)
            st.write(f"{cid}: {cname} ({difficulty_label}) — прогноз рейтинга {score:.2f}")

        # Вывод персонального маршрута
        st.subheader("Рекомендуемый путь обучения:")
        try:
            top_course_ids = [cid for cid, _ in top]
            subgraph = course_graph.subgraph(top_course_ids)

            if nx.is_directed_acyclic_graph(subgraph):
                sorted_path = list(nx.topological_sort(subgraph))
                if len(sorted_path) >= 2:
                    for i in range(len(sorted_path) - 1):
                        cname1 = course_features.loc[sorted_path[i], "CourseName"]
                        cname2 = course_features.loc[sorted_path[i+1], "CourseName"]
                        st.write(f"{cname1} → {cname2}")
                else:
                    st.write("Недостаточно данных для построения маршрута.")
            else:
                st.write("Курсы не образуют направленный маршрут.")
        except Exception as e:
            st.write("Не удалось построить маршрут обучения.")
            st.error(str(e))

    st.sidebar.markdown("---")
    st.sidebar.write("Версия модели: 4.0")
    st.sidebar.write(f"Дата: {pd.Timestamp.now().date()}")

if __name__ == "__main__":
    main()
