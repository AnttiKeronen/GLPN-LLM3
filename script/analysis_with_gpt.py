import pandas as pd
import json
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import traceback


client = OpenAI(api_key="api key")



MAX_WORKERS = 10
RETRIES = 3

def analyze(text):
    prompt = f"""
    Classify the following social media post as REAL (0) or FAKE (1).

    Post:
    {text}

    Respond ONLY in raw JSON:
    {{
        "analysis": 0 or 1,
        "prob": probability between 0 and 1
    }}
    """

    for attempt in range(RETRIES):
        try:
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            msg = r.choices[0].message.content.strip()

            if msg.startswith("```"):
                msg = msg.strip("`").replace("json", "").strip()

            return json.loads(msg)

        except Exception as e:
            print(f"Retry {attempt+1}/{RETRIES} error: {e}")
            time.sleep(1.0)

    return {"analysis": 0, "prob": 0.0}


def make_pseudo_labels(dataset):
    base = f"dataset/{dataset}"
    file = f"{base}/dataforGCN_test.csv"
    out = f"{base}/{dataset}_analysis_results.csv"

    df = pd.read_csv(file)

    # Automatically detect correct text column
    if "post_text" in df.columns:
        text_column = "post_text"
    elif "text" in df.columns:
        text_column = "text"
    else:
        raise ValueError("❌ Ei löytynyt tekstikolumnia (post_text tai text).")

    print(f"🚀 Aloitetaan pseudo-labelointi GPT:llä...")
    print(f"Testipostauksia yhteensä: {len(df)}")
    print(f"Käytetään tekstikenttää: {text_column}")

    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(analyze, text): i
            for i, text in enumerate(df[text_column])
        }

        for future in as_completed(futures):
            try:
                res = future.result()
                results.append(res)
            except:
                print("❌ Virhe yhdellä threadilla:")
                traceback.print_exc()
                results.append({"analysis": 0, "prob": 0.0})

    pd.DataFrame(results).to_csv(out, index=False)
    print(f"✔ Valmis! Pseudo-labelit tallennettu → {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", type=str)
    args = p.parse_args()

    make_pseudo_labels(args.dataset_name)
