import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import networkx as nx
import json


def main():
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

    course_features = generate_course_data()

    COURSE_DETAILS = {
        cid: {
            "name": course_features.loc[cid, "CourseName"],
            "description": f"Курс по теме: {', '.join([t.replace('Topic_', '') for t, v in zip(course_features.columns[1:11], course_features.loc[cid, 'Topics']) if v == 1])}.",
            "duration": f"{course_features.loc[cid, 'HoursPerWeek']} ч/нед",
            "difficulty": "Новичок" if course_features.loc[cid, "DifficultyLabel"] <= 2 else ("Средний" if course_features.loc[cid, "DifficultyLabel"] == 3 else "Продвинутый"),
            "topics": [t.replace("Topic_", "") for t, v in zip(course_features.columns[1:11], course_features.loc[cid, "Topics"]) if v == 1],
            "total_hours": f"{int(course_features.loc[cid, 'HoursPerWeek'] * 4)} часов в месяц"
        }
        for cid in course_features.index
    }

    def generate_synthetic_users(num_users=200, course_features=None, seed=24):
        np.random.seed(seed)
        training_data = []
        topics = [col for col in course_features.columns if col.startswith("Topic_")]
        level_map = {"Ничего не знаю":1, "Начинающий":2, "Средний":3, "Работаю в этой сфере":4, "Профи":5}

        for i in range(num_users):
            level = np.random.choice(list(level_map.keys()))
            weekly_hours = np.random.randint(1, 41)
            months = np.random.randint(1, 13)
            pref_topics = list(np.random.choice([t.replace("Topic_", "") for t in topics], size=np.random.randint(1,4), replace=False))

            user_vec = np.zeros(len(topics) + 3)
            for topic in pref_topics:
                idx = topics.index(f"Topic_{topic}")
                user_vec[idx] = 1
            user_vec[-3] = level_map[level]
            user_vec[-2] = weekly_hours
            user_vec[-1] = months

            for cid in course_features.index:
                course_vec = course_features.loc[cid, topics + ["Difficulty", "HoursPerWeek"]].values.astype(float)
                features = np.concatenate([user_vec, course_vec])
                rating = np.random.uniform(1, 5) + np.random.normal(0, 0.5)
                rating = max(1, min(5, rating))
                training_data.append(np.concatenate([features, [rating]]))

        columns = [f"U_{t}" for t in topics] + ["U_Level", "U_Hours", "U_Months"] + [f"C_{t}" for t in topics] + ["C_Difficulty", "C_CourseHours"] + ["Rating"]
        df_train = pd.DataFrame(training_data, columns=columns)
        return df_train

    if 'ml_model' not in st.session_state:
        df_train = generate_synthetic_users(course_features=course_features)
        X = df_train.drop(columns=["Rating"]).values
        y = df_train["Rating"].values
        ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
        ml_model.fit(X, y)
        st.session_state['ml_model'] = ml_model
    else:
        ml_model = st.session_state['ml_model']

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

    USERS_FILE = "users.json"

    def load_users():
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save_users(users):
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2, ensure_ascii=False)

    users = load_users()

    def get_difficulty_label(value):
        if value <= 2:
            return "Новичок"
        elif value == 3:
            return "Средний"
        else:
            return "Продвинутый"

    st.title("Персонализированные IT-учебные планы")
    username = st.text_input("Введите имя пользователя:")
    if username:
        if username not in users:
            users[username] = {"completed": []}
            save_users(users)
        st.write(f"Здравствуйте, {username}!")

        status = st.selectbox("Ваш статус:", ["Школьник", "Студент", "Работаю", "Другое"], key="status")
        level = st.selectbox("Уровень знаний:", ["Ничего не знаю", "Начинающий", "Средний", "Работаю в этой сфере", "Профи"], key="level")
        weekly_hours = st.slider("Сколько часов в неделю вы готовы уделять обучению?", 1, 40, 5, key="hours")
        months = st.slider("Сколько месяцев вы готовы потратить на обучение?", 1, 12, 3, key="months")
        goal = st.selectbox("Какова ваша цель?", ["Для себя", "Подтянуть знания для учёбы", "Хочу устроиться на работу", "Другое"], key="goal")

        preferences = st.multiselect("Интересующие темы:", [col.replace("Topic_", "") for col in course_features.columns if col.startswith("Topic_")], key="prefs")

        completed_courses = users[username].get("completed", [])
        st.write("Вы уже прошли курсы:", completed_courses)
        mark_completed = st.multiselect("Отметьте пройденные курсы:", course_features.index.tolist(), default=completed_courses)
        if st.button("Сохранить статус пройденных курсов", key="save_completed"):
            users[username]["completed"] = mark_completed
            save_users(users)
            st.success("Статус сохранён.")

        if st.button("Рекомендовать курсы (ML-модель)", key="recommend"):
            topics = [col for col in course_features.columns if col.startswith("Topic_")]
            user_vec = np.zeros(len(topics) + 3)
            level_map = {"Ничего не знаю":1, "Начинающий":2, "Средний":3, "Работаю в этой сфере":4, "Профи":5}
            for topic in preferences:
                idx = topics.index(f"Topic_{topic}")
                user_vec[idx] = 1
            user_vec[-3] = level_map[level]
            user_vec[-2] = weekly_hours
            user_vec[-1] = months

            total_hours = weekly_hours * 4 * months
            if total_hours < 100:
                max_diff = 2
            elif total_hours < 200:
                max_diff = 3
            else:
                max_diff = 5

            scores = []
            for cid in course_features.index:
                if cid in completed_courses:
                    continue
                diff = course_features.loc[cid, "DifficultyLabel"]
                if diff > max_diff:
                    continue
                # проверяем совпадение хотя бы одной темы
                course_topics = COURSE_DETAILS[cid]["topics"]
                if preferences and not set(preferences).intersection(set(course_topics)):
                    continue
                course_vec = course_features.loc[cid, topics + ["Difficulty", "HoursPerWeek"]].values.astype(float)
                features = np.concatenate([user_vec, course_vec])
                pred_rating = ml_model.predict([features])[0]
                scores.append((cid, pred_rating))

            # если меньше рекомендаций, покажем доступное количество
            top = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
            if not top:
                st.warning("Нет подходящих курсов по заданным критериям.")
            else:
                st.subheader("Рекомендованные IT-курсы (по ML-модели):")
                for cid, score in top:
                    details = COURSE_DETAILS.get(cid, {})
                    st.markdown(f"### {details.get('name', cid)}")
                    st.write(f"**Описание:** {details.get('description', 'Нет описания.')}")
                    st.write(f"**Уровень:** {details.get('difficulty', 'Неизвестно')}")
                    st.write(f"**Темы:** {', '.join(details.get('topics', []))}")
                    st.write(f"**Нагрузка:** {details.get('duration', '?')}")
                    st.write(f"**Общая нагрузка:** {details.get('total_hours', '?')}")
                    st.write(f"**Прогноз рейтинга:** {score:.2f}")
                    st.markdown("---")

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
        st.sidebar.write("Версия модели: 5.0")
        st.sidebar.write(f"Дата: {pd.Timestamp.now().date()}")
    else:
        st.info("Пожалуйста, введите имя пользователя для начала.")


if __name__ == "__main__":
    main()
