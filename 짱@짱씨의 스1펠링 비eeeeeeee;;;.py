import os
import time
import random
import sys

# ----------------------------
# 단어 목록
# ----------------------------
word_sets = {
    "Easy": ["apple", "banana", "school", "computer", "rainbow",
             "friend", "orange", "music", "water", "planet"],
    "Medium": ["beautiful", "mountain", "elephant", "adventure", "language",
               "universe", "chocolate", "astronaut", "history", "butterfly"],
    "Hard": ["encyclopedia", "psychology", "architecture", "biochemistry",
             "transformation", "constitution", "circumference",
             "photosynthesis", "magnificent", "sustainability"],
    "Extreme": ["antidisestablishmentarianism", "floccinaucinihilipilification",
                "pseudopseudohypoparathyroidism", "supercalifragilisticexpialidocious",
                "hippopotomonstrosesquippedaliophobia", "pneumonoultramicroscopicsilicovolcanoconiosis",
                "honorificabilitudinitatibus", "thyroparathyroidectomized",
                "incomprehensibilities", "deinstitutionalization"]
}

# ----------------------------
# 유틸 함수
# ----------------------------
def clear():
    os.system("cls" if os.name == "nt" else "clear")

def print_banner():
    clear()
    print("🐝================================================🐝")
    print("           jangjang's Spelling Bee")
    print("🐝================================================🐝\n")
    time.sleep(0.4)

def slow_print(text, delay=0.03):
    for ch in text:
        print(ch, end='', flush=True)
        time.sleep(delay)
    print()

def select_difficulty():
    while True:
        print_banner()
        print("난이도를 선택하세요:")
        print("[1] Easy")
        print("[2] Medium")
        print("[3] Hard")
        print("[4] Extreme\n")
        choice = input("👉 난이도 번호 입력: ").strip()
        mapping = {"1": "Easy", "2": "Medium", "3": "Hard", "4": "Extreme"}
        if choice in mapping:
            return mapping[choice]
        else:
            print("잘못된 입력입니다. 다시 선택하세요.")
            time.sleep(1)

def flush_input():
    """Enter 버퍼 제거 (Pydroid 포함 안전 처리)"""
    try:
        import termios, tty, select
        while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
    except ImportError:
        # Windows/Pydroid: input() 호출 전에 남은 줄 없으면 무시
        pass

def safe_input(prompt=""):
    """사용자 입력 처리 안전하게"""
    try:
        return input(prompt)
    except EOFError:
        return ""

def game_round(words, difficulty):
    score = 0
    random.shuffle(words)
    for i, word in enumerate(words, start=1):
        clear()
        print(f"🐝 {difficulty} 모드 | 라운드 {i}/{len(words)} 🐝\n")
        time.sleep(1)

        print("단어가 표시됩니다... 준비하세요!")
        time.sleep(1)
        # 긴 단어도 한 번에 보이도록 표시
        print(word)
        time.sleep(2)
        clear()

        print(f"[Round {i}] 단어를 철자하세요:")

        flush_input()  # Enter 버퍼 제거
        answer = safe_input("👉 ").strip().lower()

        if answer == word:
            print("✅ 정답! 완벽했어요!")
            score += 1
            time.sleep(1)
        else:
            print(f"❌ 오답! 정답은: {word}")
            time.sleep(1.5)
            break  # 오답이면 라운드 종료
    return score

def show_result(score, total):
    clear()
    print_banner()
    print(f"🎯 최종 점수: {score}/{total}\n")
    if score == total:
        print("👑 완벽! 당신은 언어의 제왕이에요!")
    elif score >= total // 2:
        print("🔥 훌륭해요! 거의 다 맞췄어요!")
    else:
        print("😂 아쉽네요! 다음엔 더 잘할 수 있어요!")
    print("\n🐝 jangjang's Spelling Bee 종료 🐝\n")

# ----------------------------
# 실행
# ----------------------------
if __name__ == "__main__":
    print_banner()
    slow_print("Welcome to jangjang's Spelling Bee Challenge!", 0.04)
    safe_input("\n시작하려면 Enter를 누르세요...")

    difficulty = select_difficulty()
    words = word_sets[difficulty]
    total = len(words)

    score = game_round(words, difficulty)
    show_result(score, total)