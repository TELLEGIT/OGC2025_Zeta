import os
import json
import subprocess

# 테스트 이름과 설정
test_name = "test4"
timelimit = 60
problem_dir = "/Users/telle/Desktop/ogc2025/exercise_problems"
problem_files = [f"prob{i}.json" for i in range(1, 11)]

# 저장 경로 설정
log_dir = os.path.join("test_logs", test_name)
result_dir = os.path.join("test_results", test_name)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

# 알고리즘 파일 위치
algorithm_path = "/Users/telle/Desktop/ogc2025/baseline_20250601/test_algorithm.py"

for prob_file in problem_files:
    prob_path = os.path.join(problem_dir, prob_file)
    prob_name = os.path.splitext(prob_file)[0]  # prob1, prob2, ...

    log_file_path = os.path.join(log_dir, f"{test_name}_{prob_name}_log.txt")
    result_file_path = os.path.join(result_dir, f"{test_name}_{prob_name}_results.json")

    print(f"▶ Running {prob_name}...")

    # 실행 명령어: stdout을 로그로, results.json을 결과 파일로 저장
    try:
        result = subprocess.run(
            ["python", algorithm_path, test_name, prob_path, str(timelimit)],
            capture_output=True,
            text=True,
            timeout=timelimit + 5
        )

        # 로그 저장
        with open(log_file_path, "w") as log_f:
            log_f.write(result.stdout)
            if result.stderr:
                log_f.write("\n\n[stderr]\n" + result.stderr)

        # results.json 파일 이동
        if os.path.exists("results.json"):
            with open("results.json", "r") as rf:
                result_json = json.load(rf)
            with open(result_file_path, "w") as wf:
                json.dump(result_json, wf, indent=2)
            os.remove("results.json")

        print(f"{prob_name} 완료")

    except Exception as e:
        print(f"{prob_name} 실패: {e}")