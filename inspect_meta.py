import json
import argparse


def inspect_meta(index_dir: str, chunk_lines: int = 5, sentence_lines: int = 10):
    """Print first n lines from meta_chunk.jsonl and first m lines from meta_sentence.jsonl."""
    
    chunk_path = f"{index_dir}/meta_chunk.jsonl"
    sentence_path = f"{index_dir}/meta_sentence.jsonl"
    
    print(f"{'='*80}")
    print(f"CHUNKS (first {chunk_lines} lines from {chunk_path})")
    print(f"{'='*80}\n")
    
    try:
        with open(chunk_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= chunk_lines:
                    break
                data = json.loads(line)
                print(f"[Chunk {i}]")
                print(f"  {data.get('text', '')}")
                print()
    except FileNotFoundError:
        print(f"File not found: {chunk_path}\n")
    
    print(f"{'='*80}")
    print(f"SENTENCES (first {sentence_lines} lines from {sentence_path})")
    print(f"{'='*80}\n")
    
    try:
        with open(sentence_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= sentence_lines:
                    break
                data = json.loads(line)
                print(f"[Sentence {i}]")
                print(f"  {data.get('text', '')}")
                print()
    except FileNotFoundError:
        print(f"File not found: {sentence_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect meta_chunk and meta_sentence files")
    parser.add_argument("--index_dir", type=str, default="./index", help="Index directory")
    parser.add_argument("--chunks", type=int, default=5, help="Number of chunks to display")
    parser.add_argument("--sentences", type=int, default=10, help="Number of sentences to display")
    
    args = parser.parse_args()
    inspect_meta(args.index_dir, args.chunks, args.sentences)
