import streamlit as st
from PyPDF2 import PdfReader
from groq import Groq
import json
import os
import hashlib
from datetime import datetime, date, time
import pandas as pd

# 🔽 Groq API 키 로드
GROQ_API_KEY = st.secrets["GROQ_API_KEY"] # Streamlit Secrets에서 API 키 로드

client = Groq(api_key=GROQ_API_KEY)

USER_DATA_PATH = "users.json" # 사용자 계정 정보 저장 경로

# --- 비밀번호 해시 함수 ---
def hash_password(password):
    """비밀번호를 SHA256으로 해시합니다."""
    return hashlib.sha256(password.encode()).hexdigest()

# --- 사용자 DB 함수 ---
def load_users():
    """사용자 계정 정보를 로드합니다."""
    if os.path.exists(USER_DATA_PATH):
        with open(USER_DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_users(users):
    """사용자 계정 정보를 저장합니다."""
    with open(USER_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False)

# --- 회원가입 ---
def signup(username, password):
    """새로운 사용자를 등록합니다."""
    users = load_users()
    if username in users:
        return False, "이미 존재하는 사용자입니다."
    users[username] = hash_password(password)
    save_users(users)
    return True, "회원가입 성공! 로그인 해주세요."

# --- 로그인 ---
def login(username, password):
    """사용자 로그인을 처리합니다."""
    users = load_users()
    if username not in users:
        return False, "존재하지 않는 사용자입니다."
    if users[username] != hash_password(password):
        return False, "비밀번호가 틀렸습니다."
    return True, "로그인 성공!"

# --- 유틸 함수들 ---
def extract_text_from_pdf(file):
    """PDF 파일에서 텍스트를 추출합니다."""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# --- 사용자별 파일 경로 생성 함수 ---
def get_chat_save_path(username):
    """사용자별 채팅 기록 파일 경로를 반환합니다."""
    return f"chat_history_{username}.json"

def get_schedule_save_path(username):
    """사용자별 일정 파일 경로를 반환합니다."""
    return f"schedules_{username}.json"

def get_persona_save_path(username):
    """사용자별 AI 페르소나 설정 파일 경로를 반환합니다."""
    return f"ai_persona_{username}.json"

# --- JSON 저장/로드 함수 (리스트, 딕셔너리 모두 사용 가능) ---
def save_json(path, data):
    """JSON 데이터를 지정된 경로에 저장합니다."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def load_json(path):
    """JSON 리스트 데이터를 지정된 경로에서 로드합니다. 파일이 없으면 빈 리스트를 반환합니다."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def load_json_dict(path):
    """JSON 딕셔너리 데이터를 지정된 경로에서 로드합니다. 파일이 없으면 빈 딕셔너리를 반환합니다."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# --- Groq API 호출 함수 (AI 페르소나 설정 반영) ---
def call_groq_api(user_msg, doc_summary="", max_tokens=512):
    """
    Groq API를 호출하여 AI 응답을 받습니다.
    사용자별 AI 페르소나 설정을 system_msg에 반영합니다.
    """
    current_username = st.session_state.username
    persona_settings = load_json_dict(get_persona_save_path(current_username)) # 사용자별 페르소나 로드

    # 기본 시스템 메시지
    system_msg_parts = ["당신은 사용자의 개인 비서입니다. 모든 답변은 한국어로 해주세요."]

    # 페르소나 설정 반영
    if persona_settings:
        if persona_settings.get("tone"):
            system_msg_parts.append(f"말투는 '{persona_settings['tone']}'으로 유지해주세요.")
        if persona_settings.get("mind"):
            system_msg_parts.append(f"다음과 같은 마인드를 가지고 답변해주세요: {persona_settings['mind']}")
        if persona_settings.get("focus_areas"):
            system_msg_parts.append(f"특히 다음 주제에 중점을 두어 답변해주세요: {persona_settings['focus_areas']}")
    
    # 문서 요약이 있다면 시스템 메시지에 추가
    if doc_summary:
        system_msg_parts.append(f"제공된 문서 요약을 참고하여 답변해주세요. 문서 요약: {doc_summary}")

    final_system_msg = " ".join(system_msg_parts)
    
    messages = [
        {"role": "system", "content": final_system_msg},
        {"role": "user", "content": user_msg},
    ]
    
    # temperature 값은 persona_settings에서 가져오거나 기본값 사용
    temp = persona_settings.get("temperature", 0.5)

    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
        temperature=temp,
        max_completion_tokens=max_tokens,
        top_p=1,
        stream=False
    )
    return completion.choices[0].message.content

# --- 세션 상태 초기화 (로그인 상태에 따라 데이터 로드) ---
if "login_status" not in st.session_state:
    st.session_state.login_status = False
    st.session_state.username = ""
    st.session_state.chat_history = []
    st.session_state.doc_summary = ""
    st.session_state.schedules = []
    st.session_state.ai_persona_settings = {} # AI 페르소나 설정 초기화

else:
    # 로그인 상태가 있다면 해당 사용자의 데이터를 로드
    if st.session_state.login_status and st.session_state.username:
        st.session_state.chat_history = load_json(get_chat_save_path(st.session_state.username))
        st.session_state.schedules = load_json(get_schedule_save_path(st.session_state.username))
        st.session_state.ai_persona_settings = load_json_dict(get_persona_save_path(st.session_state.username))
    else: # 로그인되지 않은 상태 (혹시 모를 경우를 대비하여 세션 상태 초기화)
        st.session_state.chat_history = []
        st.session_state.doc_summary = ""
        st.session_state.schedules = []
        st.session_state.ai_persona_settings = {}

# --- 로그인 UI 및 처리 ---
def login_ui():
    """사용자 로그인 및 회원가입 UI를 렌더링합니다."""
    st.title("🔐 사용자 로그인 / 회원가입")
    menu = st.radio("원하는 작업을 선택하세요.", ["로그인", "회원가입"])

    if menu == "로그인":
        st.subheader("로그인")
        if "login_message" not in st.session_state:
            st.session_state.login_message = ""

        with st.form("login_form"):
            username = st.text_input("아이디", placeholder="아이디를 입력하세요", key="login_user")
            password = st.text_input("비밀번호", type="password", placeholder="비밀번호를 입력하세요", key="login_pw")
            submitted = st.form_submit_button("로그인")

        if submitted:
            success, msg = login(username, password)
            if success:
                st.session_state.login_status = True
                st.session_state.username = username
                # 로그인 성공 시, 해당 사용자의 데이터를 로드
                st.session_state.chat_history = load_json(get_chat_save_path(username))
                st.session_state.schedules = load_json(get_schedule_save_path(username))
                st.session_state.ai_persona_settings = load_json_dict(get_persona_save_path(username))
                st.session_state.login_message = ""
                st.success(f"{st.session_state.username}님 로그인 성공!")
                st.rerun() # 로그인 성공 후 앱 재실행
            else:
                st.session_state.login_message = msg
                st.error(st.session_state.login_message)

    elif menu == "회원가입":
        st.subheader("회원가입")
        with st.form("signup_form"):
            new_username = st.text_input("아이디", key="signup_user", placeholder="사용할 아이디를 입력하세요")
            new_password = st.text_input("비밀번호", type="password", key="signup_pw", placeholder="비밀번호를 입력하세요")
            new_password2 = st.text_input("비밀번호 확인", type="password", key="signup_pw2", placeholder="비밀번호를 다시 입력하세요")
            submitted = st.form_submit_button("회원가입")
            if submitted:
                if new_password != new_password2:
                    st.error("비밀번호가 일치하지 않습니다.")
                elif not new_username or not new_password:
                    st.error("아이디와 비밀번호를 모두 입력하세요.")
                else:
                    success, msg = signup(new_username, new_password)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)

# --- Helper function for parsing time strings (MODIFIED) ---
def parse_schedule_time(time_str):
    """
    시간 문자열을 datetime.time 객체로 파싱합니다.
    이미 datetime.time 객체인 경우 그대로 반환합니다.
    """
    if isinstance(time_str, time):
        return time_str
    if not isinstance(time_str, str):
        # 예상치 못한 타입이면 None 반환 (또는 에러 처리)
        return None
    
    # Try parsing common time formats, prioritizing more specific ones
    formats = ["%H:%M:%S.%f", "%H:%M:%S", "%H:%M"]
    for fmt in formats:
        try:
            return datetime.strptime(time_str, fmt).time()
        except ValueError:
            continue
    
    # If all parsing fails, return None (or original string with warning)
    st.warning(f"Warning: Could not parse time string '{time_str}'. This might lead to display issues.")
    return None # None을 반환하여 데이터 에디터가 빈 값으로 처리하도록 유도

# --- Existing app UI (part shown after login) ---
def app_main():
    """로그인 후 메인 앱 UI를 렌더링합니다."""
    current_username = st.session_state.username # 현재 사용자 이름 가져오기
    
    st.title(f"🙋‍♂️ 안녕하세요, {current_username}님!")
    if st.button("로그아웃"):
        st.session_state.login_status = False
        st.session_state.username = ""
        # 로그아웃 시 모든 세션 데이터 초기화
        st.session_state.chat_history = []
        st.session_state.doc_summary = ""
        st.session_state.schedules = []
        st.session_state.ai_persona_settings = {}
        st.rerun()

    # 'AI 비서 설정' 탭을 포함한 메뉴 선택
    tab = st.sidebar.selectbox("메뉴 선택", ["문서 요약 & 상담", "일정 관리", "AI 비서 설정"])

    if tab == "문서 요약 & 상담":
        st.header("📄 문서 요약 & AI 상담")

        uploaded_file = st.file_uploader("📂 PDF 문서 업로드", type=["pdf"])
        if uploaded_file:
            with st.spinner("문서 텍스트 추출 중..."):
                text = extract_text_from_pdf(uploaded_file)
            with st.spinner("문서 요약 중..."):
                st.session_state.doc_summary = call_groq_api(
                    user_msg=f"아래 문서를 간결하게 3~4문장으로 요약해 주세요.",
                    doc_summary=text[:3000],
                    max_tokens=512
                )
            st.subheader("📝 문서 요약")
            st.write(st.session_state.doc_summary)

        st.subheader("🤖 AI 상담")
        user_input = st.text_input("질문을 입력하세요", key="chat_input")

        col1, col2 = st.columns([1,1])
        with col1:
            if st.button("질문 제출") and user_input:
                with st.spinner("답변 생성 중..."):
                    answer = call_groq_api(user_msg=user_input, doc_summary=st.session_state.doc_summary)

                st.session_state.chat_history.append({"질문": user_input, "답변": answer})
                save_json(get_chat_save_path(current_username), st.session_state.chat_history)
                st.rerun()

        with col2:
            clear_button_clicked = st.button("대화 기록 초기화")
            
        if clear_button_clicked:
            st.session_state.chat_history = []
            chat_file_path = get_chat_save_path(current_username)
            if os.path.exists(chat_file_path):
                os.remove(chat_file_path)
            
            full_width_message_placeholder = st.empty()
            full_width_message_placeholder.success("대화 기록이 초기화되었습니다. 페이지를 새로고침 해주세요.")
            full_width_message_placeholder.empty() # 메시지 제거
            st.rerun() # 변경사항 즉시 반영 

        if st.session_state.chat_history:
            st.subheader("🗨 대화 기록")
            for chat in reversed(st.session_state.chat_history):
                st.markdown(f"**Q:** {chat['질문']}")
                st.markdown(f"**A:** {chat['답변']}")
                st.markdown("---")

    elif tab == "일정 관리":
        st.header("📅 일정 관리")

        # 세션 상태 초기화 (한 번만)
        if "schedule_date" not in st.session_state:
            st.session_state.schedule_date = date.today()
        if "schedule_time" not in st.session_state:
            st.session_state.schedule_time = datetime.now().time()
        if "schedule_event" not in st.session_state:
            st.session_state.schedule_event = ""

        # 일정 추가 UI
        with st.form("add_schedule_form", clear_on_submit=False):
            st.subheader("새 일정 추가")

            st.session_state.schedule_date = st.date_input("날짜 선택", value=st.session_state.schedule_date)
            st.session_state.schedule_time = st.time_input("시간 선택", value=st.session_state.schedule_time)
            st.session_state.schedule_event = st.text_input("일정 내용", value=st.session_state.schedule_event, placeholder="예: 오후 3시 회의, 저녁 식사 약속")
            add_button = st.form_submit_button("일정 추가")

            if add_button:
                if st.session_state.schedule_event.strip():
                    st.session_state.schedules.append({
                        "date": str(st.session_state.schedule_date),
                        "time": st.session_state.schedule_time.strftime("%H:%M"),
                        "event": st.session_state.schedule_event
                    })
                    save_json(get_schedule_save_path(current_username), st.session_state.schedules)
                    st.success("일정이 추가되었습니다!")
                    st.session_state.schedule_event = ""
                    st.rerun()
                else:
                    st.warning("일정 내용을 입력해주세요.")

        st.subheader("등록된 일정 목록 (편집 가능)")

        if st.session_state.schedules:
            schedule_data_for_df = []
            for i, s in enumerate(st.session_state.schedules):
                try:
                    parsed_date = datetime.strptime(s['date'], "%Y-%m-%d").date()
                    parsed_time = parse_schedule_time(s['time'])
                    schedule_data_for_df.append({
                        "날짜": parsed_date,
                        "시간": parsed_time,
                        "일정 내용": s['event'],
                        "_original_id": f"{s['date']}_{s['time']}_{s['event']}_{i}"
                    })
                except (ValueError, TypeError) as e:
                    st.warning(f"Warning: Error parsing schedule data for Data Editor - {e}. Skipping entry: {s}")
                    schedule_data_for_df.append({
                        "날짜": s['date'],
                        "시간": s['time'],
                        "일정 내용": s['event'],
                        "_original_id": f"{s['date']}_{s['time']}_{s['event']}_{i}"
                    })

            df = pd.DataFrame(schedule_data_for_df)
            
            if not df.empty:
                df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce').dt.date
                df['시간'] = df['시간'].apply(lambda x: parse_schedule_time(x))
                df['sort_key'] = df.apply(
                    lambda row: datetime.combine(row['날짜'], row['시간'] if row['시간'] else time(0,0)) if pd.notna(row['날짜']) else pd.NaT,
                    axis=1
                )
                df = df.sort_values(by='sort_key', na_position='first').drop(columns='sort_key').reset_index(drop=True)

            edited_df = st.data_editor(
                df,
                column_config={
                    "날짜": st.column_config.DateColumn("날짜", format="YYYY-MM-DD"),
                    "시간": st.column_config.TimeColumn("시간", format="HH:mm"),
                    "일정 내용": st.column_config.TextColumn("일정 내용", width="large"),
                    "_original_id": None
                },
                hide_index=True,
                num_rows="dynamic",
                use_container_width=True,
                key="schedule_data_editor"
            )

            new_schedules_from_editor = []
            for _, row in edited_df.iterrows():
                date_val = row["날짜"]
                time_val = row["시간"]
                event_val = row["일정 내용"]

                if not event_val or str(event_val).strip() == "":
                    continue

                date_str = ""
                if isinstance(date_val, date):
                    date_str = date_val.strftime("%Y-%m-%d")
                elif isinstance(date_val, str):
                    date_str = date_val
                
                time_str = ""
                if isinstance(time_val, time):
                    time_str = time_val.strftime("%H:%M")
                elif isinstance(time_val, str):
                    time_str = time_val

                new_schedules_from_editor.append({
                    "date": date_str,
                    "time": time_str,
                    "event": event_val
                })

            comparable_new_schedules = sorted([
                (s['date'], s['time'], s['event']) for s in new_schedules_from_editor
            ])
            comparable_existing_schedules = sorted([
                (s['date'], s['time'], s['event']) for s in st.session_state.schedules
            ])

            if comparable_new_schedules != comparable_existing_schedules:
                st.session_state.schedules = new_schedules_from_editor
                save_json(get_schedule_save_path(current_username), st.session_state.schedules)
                st.success("일정이 업데이트되었습니다!")
                st.rerun()

        else:
            st.info("아직 등록된 일정이 없습니다. 일정을 추가하면 AI가 도와드릴 수 있어요!")

        st.subheader("🤖 AI 일정 도우미")

        if st.session_state.schedules:
            schedule_text = ""
            sorted_schedules = sorted(st.session_state.schedules, key=lambda x: (x['date'], x['time']))
            
            for sched_item in sorted_schedules:
                row_date = sched_item['date']
                row_time = sched_item['time']
                row_event = sched_item['event']
                if row_event.strip():
                    schedule_text += f"- {row_date} {row_time}: {row_event}\n"

            if st.button("현재 일정 요약 및 분석 요청"):
                with st.spinner("AI가 일정을 분석 중입니다..."):
                    ai_response = call_groq_api(
                        user_msg=f"내 일정 목록:\n{schedule_text}\n\n이 일정을 바탕으로 주요 내용을 3-4문장으로 요약하고, 특이사항이나 중요한 패턴이 있다면 분석하여 알려줘.",
                        max_tokens=1024
                    )
                    st.markdown("#### ✨ AI 분석 결과:")
                    st.write(ai_response)
            
            if st.button("다음 주 추천 일정 요청"):
                with st.spinner("AI가 다음 주 일정을 추천 중입니다..."):
                    ai_recommendation = call_groq_api(
                        user_msg=f"내 기존 일정 목록:\n{schedule_text}\n\n이것을 바탕으로 다음 주(오늘 기준 7일 이내) 추천 일정을 2~3개 제안해줘. 추천하는 일정은 간단한 활동(예: 산책, 독서, 휴식 등)이 좋고, 날짜와 시간도 구체적으로 포함해서 작성해줘.",
                        max_tokens=512
                    )
                    st.markdown("#### ✨ AI 추천 일정:")
                    st.write(ai_recommendation)
        else:
            st.info("아직 등록된 일정이 없습니다. 일정을 추가하면 AI가 도와드릴 수 있어요!")

    # AI 비서 설정 탭 (변경 없음)
    elif tab == "AI 비서 설정":
        st.header("⚙️ 나만의 AI 비서 설정")
        st.write("AI 비서의 말투, 마인드, 중점적으로 생각할 주제를 설정할 수 있습니다.")

        current_persona = st.session_state.ai_persona_settings

        with st.form("persona_form"):
            tone_options = ["기본 (친절하고 일반적)", "전문적 (정보 전달 위주)", "유머러스 (재미있게)", "간결함 (짧고 핵심만)", "비판적 (분석적 사고)"]
            selected_tone = st.selectbox(
                "말투 선택:",
                options=tone_options,
                index=tone_options.index(current_persona.get("tone", tone_options[0])),
                key="ai_tone_select"
            )
            mind_input = st.text_area(
                "AI의 마인드를 설정하세요 (예: 항상 긍정적으로, 논리적으로 분석):",
                value=current_persona.get("mind", ""),
                height=100,
                key="ai_mind_input"
            )
            focus_areas_input = st.text_area(
                "AI가 중점적으로 다룰 주제를 입력하세요 (쉼표로 구분):",
                value=current_persona.get("focus_areas", ""),
                height=70,
                key="ai_focus_areas_input"
            )
            temperature_value = st.slider(
                "AI의 창의성 (Temperature):",
                min_value=0.0,
                max_value=1.0,
                value=current_persona.get("temperature", 0.5),
                step=0.1,
                help="값이 높을수록 더 창의적이고 예측 불가능한 답변을 생성합니다. 낮을수록 더 일관적이고 보수적입니다.",
                key="ai_temperature_slider"
            )

            submitted = st.form_submit_button("AI 비서 설정 저장")

            if submitted:
                new_persona_settings = {
                    "tone": selected_tone,
                    "mind": mind_input,
                    "focus_areas": focus_areas_input,
                    "temperature": temperature_value
                }
                st.session_state.ai_persona_settings = new_persona_settings
                save_json(get_persona_save_path(current_username), new_persona_settings)
                st.success("AI 비서 설정이 저장되었습니다! 다음 대화부터 적용됩니다.")
                st.rerun()

        st.subheader("현재 AI 비서 설정")
        if st.session_state.ai_persona_settings:
            st.json(st.session_state.ai_persona_settings)
        else:
            st.info("아직 AI 비서 설정이 없습니다. 위에서 설정해주세요.")


# --- 실행 ---
def main():
    """앱의 메인 진입점입니다."""
    st.set_page_config(page_title="멀티기능 AI 비서", layout="wide")
    if st.session_state.login_status:
        app_main()
    else:
        login_ui()

if __name__ == "__main__":
    main()
    
    
    