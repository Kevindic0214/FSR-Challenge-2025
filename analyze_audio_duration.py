#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audio Duration Statistics Analyzer for FSR-2025 Hakka ASR

This script analyzes the duration statistics of audio files in the training dataset,
providing insights into the temporal characteristics of the HAT-Vol2 corpus.

Usage:
    python analyze_audio_duration.py --root HAT-Vol2
    python analyze_audio_duration.py --manifest HAT-Vol2/manifests_track1/train.jsonl
    python analyze_audio_duration.py --root HAT-Vol2 --export_csv duration_stats.csv
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torchaudio
from tqdm import tqdm
import numpy as np

def get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file in seconds."""
    try:
        info = torchaudio.info(audio_path)
        duration = info.num_frames / info.sample_rate
        return duration
    except Exception as e:
        print(f"Error loading {audio_path}: {e}", file=sys.stderr)
        return 0.0

def scan_audio_files(root_dir: Path) -> List[str]:
    """Scan for all audio files in training directories."""
    audio_files = []
    
    # Look for training directories
    train_dirs = [d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("訓練_")]
    
    if not train_dirs:
        print(f"No training directories found in {root_dir}")
        return audio_files
    
    print(f"Found {len(train_dirs)} training directories:")
    for train_dir in train_dirs:
        print(f"  - {train_dir.name}")
        
        # Recursively find all wav files
        wav_files = list(train_dir.rglob("*.wav"))
        audio_files.extend([str(f) for f in wav_files])
        print(f"    Found {len(wav_files)} audio files")
    
    return audio_files

def load_from_manifest(manifest_path: Path, root_dir: Optional[Path] = None) -> List[str]:
    """Load audio file paths from JSONL manifest."""
    audio_files = []
    
    with manifest_path.open('r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                audio_path = data.get('audio', '')
                
                if not audio_path:
                    print(f"Warning: No audio path in line {line_num}")
                    continue
                
                # Handle relative paths
                audio_path_obj = Path(audio_path)
                if not audio_path_obj.is_absolute() and root_dir:
                    audio_path_obj = root_dir / audio_path_obj
                
                if not audio_path_obj.exists():
                    print(f"Warning: Audio file not found: {audio_path_obj}")
                    continue
                
                audio_files.append(str(audio_path_obj))
                
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON in line {line_num}")
                continue
    
    return audio_files

def extract_speaker_info(audio_path: str) -> Dict[str, str]:
    """Extract speaker and dialect information from audio path."""
    path = Path(audio_path)
    
    # Extract speaker ID from parent directory name
    speaker_id = path.parent.name if path.parent.name != path.parent.parent.name else "unknown"
    
    # Extract dialect from training directory name
    dialect = "unknown"
    for part in path.parts:
        if part.startswith("訓練_"):
            if "大埔腔" in part:
                dialect = "大埔腔"
            elif "詔安腔" in part:
                dialect = "詔安腔"
            break
    
    # Extract gender and group from speaker ID
    gender = "unknown"
    group = "unknown"
    if len(speaker_id) >= 2:
        group = speaker_id[:2]  # DF, DM, ZF, ZM
        if len(speaker_id) >= 3:
            gender_char = speaker_id[2]
            if gender_char in ['F', 'f']:
                gender = "女"
            elif gender_char in ['M', 'm']:
                gender = "男"
    
    return {
        "speaker_id": speaker_id,
        "dialect": dialect,
        "gender": gender,
        "group": group
    }

def analyze_durations(audio_files: List[str], show_progress: bool = True) -> Dict:
    """Analyze duration statistics for a list of audio files."""
    durations = []
    speaker_stats = defaultdict(list)
    dialect_stats = defaultdict(list)
    group_stats = defaultdict(list)
    
    # Progress bar setup
    if show_progress:
        pbar = tqdm(audio_files, desc="Analyzing audio files", ncols=100)
    else:
        pbar = audio_files
    
    skipped = 0
    
    for audio_path in pbar:
        duration = get_audio_duration(audio_path)
        
        if duration <= 0:
            skipped += 1
            continue
        
        durations.append(duration)
        
        # Extract metadata
        info = extract_speaker_info(audio_path)
        speaker_stats[info["speaker_id"]].append(duration)
        dialect_stats[info["dialect"]].append(duration)
        group_stats[info["group"]].append(duration)
        
        if show_progress and len(durations) % 100 == 0:
            pbar.set_postfix({
                'processed': len(durations),
                'avg_dur': f"{np.mean(durations):.1f}s"
            })
    
    if not durations:
        print("No valid audio files found!")
        return {}
    
    # Calculate overall statistics
    durations_np = np.array(durations)
    
    stats = {
        'total_files': len(durations),
        'skipped_files': skipped,
        'total_hours': np.sum(durations_np) / 3600,
        'min_duration': np.min(durations_np),
        'max_duration': np.max(durations_np),
        'mean_duration': np.mean(durations_np),
        'median_duration': np.median(durations_np),
        'std_duration': np.std(durations_np),
        'percentiles': {
            '5%': np.percentile(durations_np, 5),
            '25%': np.percentile(durations_np, 25),
            '75%': np.percentile(durations_np, 75),
            '95%': np.percentile(durations_np, 95),
        },
        'speaker_stats': {},
        'dialect_stats': {},
        'group_stats': {}
    }
    
    # Calculate per-speaker statistics
    for speaker, speaker_durations in speaker_stats.items():
        if speaker_durations:
            speaker_np = np.array(speaker_durations)
            stats['speaker_stats'][speaker] = {
                'count': len(speaker_durations),
                'total_hours': np.sum(speaker_np) / 3600,
                'mean_duration': np.mean(speaker_np),
                'std_duration': np.std(speaker_np)
            }
    
    # Calculate per-dialect statistics  
    for dialect, dialect_durations in dialect_stats.items():
        if dialect_durations:
            dialect_np = np.array(dialect_durations)
            stats['dialect_stats'][dialect] = {
                'count': len(dialect_durations),
                'total_hours': np.sum(dialect_np) / 3600,
                'mean_duration': np.mean(dialect_np),
                'std_duration': np.std(dialect_np)
            }
    
    # Calculate per-group statistics
    for group, group_durations in group_stats.items():
        if group_durations:
            group_np = np.array(group_durations)
            stats['group_stats'][group] = {
                'count': len(group_durations),
                'total_hours': np.sum(group_np) / 3600,
                'mean_duration': np.mean(group_np),
                'std_duration': np.std(group_np)
            }
    
    return stats

def print_statistics(stats: Dict):
    """Print formatted statistics."""
    print("\n" + "="*60)
    print("📊 AUDIO DURATION STATISTICS")
    print("="*60)
    
    print(f"\n📈 Overall Statistics:")
    print(f"  Total files:        {stats['total_files']:,}")
    if stats['skipped_files'] > 0:
        print(f"  Skipped files:      {stats['skipped_files']:,}")
    print(f"  Total duration:     {stats['total_hours']:.2f} hours")
    print(f"  Min duration:       {stats['min_duration']:.2f} seconds")
    print(f"  Max duration:       {stats['max_duration']:.2f} seconds") 
    print(f"  Mean duration:      {stats['mean_duration']:.2f} seconds")
    print(f"  Median duration:    {stats['median_duration']:.2f} seconds")
    print(f"  Std deviation:      {stats['std_duration']:.2f} seconds")
    
    print(f"\n📊 Duration Percentiles:")
    for percentile, value in stats['percentiles'].items():
        print(f"  {percentile:>3}: {value:>8.2f} seconds")
    
    # Dialect statistics
    if stats['dialect_stats']:
        print(f"\n🗣️ Statistics by Dialect:")
        for dialect, dialect_stats in stats['dialect_stats'].items():
            print(f"  {dialect}:")
            print(f"    Files:      {dialect_stats['count']:,}")
            print(f"    Duration:   {dialect_stats['total_hours']:.2f} hours")
            print(f"    Mean:       {dialect_stats['mean_duration']:.2f} seconds")
    
    # Group statistics
    if stats['group_stats']:
        print(f"\n👥 Statistics by Group:")
        for group, group_stats in sorted(stats['group_stats'].items()):
            if group != "unknown":
                print(f"  {group}:")
                print(f"    Files:      {group_stats['count']:,}")
                print(f"    Duration:   {group_stats['total_hours']:.2f} hours")
                print(f"    Mean:       {group_stats['mean_duration']:.2f} seconds")
    
    # Top speakers by duration
    if stats['speaker_stats']:
        top_speakers = sorted(
            [(spk, data) for spk, data in stats['speaker_stats'].items()],
            key=lambda x: x[1]['total_hours'],
            reverse=True
        )[:10]
        
        print(f"\n🎤 Top 10 Speakers by Total Duration:")
        for i, (speaker, speaker_stats) in enumerate(top_speakers, 1):
            print(f"  {i:2d}. {speaker}: {speaker_stats['total_hours']:.2f}h "
                  f"({speaker_stats['count']} files, avg: {speaker_stats['mean_duration']:.1f}s)")

def export_to_csv(stats: Dict, csv_path: str):
    """Export detailed statistics to CSV."""
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write overall stats
        writer.writerow(['Category', 'Metric', 'Value'])
        writer.writerow(['Overall', 'Total Files', stats['total_files']])
        writer.writerow(['Overall', 'Total Hours', f"{stats['total_hours']:.2f}"])
        writer.writerow(['Overall', 'Min Duration (s)', f"{stats['min_duration']:.2f}"])
        writer.writerow(['Overall', 'Max Duration (s)', f"{stats['max_duration']:.2f}"])
        writer.writerow(['Overall', 'Mean Duration (s)', f"{stats['mean_duration']:.2f}"])
        writer.writerow(['Overall', 'Median Duration (s)', f"{stats['median_duration']:.2f}"])
        writer.writerow(['Overall', 'Std Duration (s)', f"{stats['std_duration']:.2f}"])
        
        # Write percentiles
        for percentile, value in stats['percentiles'].items():
            writer.writerow(['Percentile', percentile, f"{value:.2f}"])
        
        # Write dialect stats
        writer.writerow([])  # Empty row
        writer.writerow(['Dialect', 'Files', 'Hours', 'Mean Duration (s)'])
        for dialect, dialect_stats in stats['dialect_stats'].items():
            writer.writerow([
                dialect,
                dialect_stats['count'],
                f"{dialect_stats['total_hours']:.2f}",
                f"{dialect_stats['mean_duration']:.2f}"
            ])
        
        # Write speaker stats
        writer.writerow([])  # Empty row
        writer.writerow(['Speaker', 'Files', 'Hours', 'Mean Duration (s)', 'Std Duration (s)'])
        for speaker, speaker_stats in sorted(stats['speaker_stats'].items()):
            writer.writerow([
                speaker,
                speaker_stats['count'],
                f"{speaker_stats['total_hours']:.2f}",
                f"{speaker_stats['mean_duration']:.2f}",
                f"{speaker_stats['std_duration']:.2f}"
            ])
    
    print(f"\n💾 Detailed statistics exported to: {csv_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze audio duration statistics for FSR-2025 Hakka ASR training data"
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--root", type=Path, 
                            help="HAT-Vol2 root directory (scans training directories)")
    input_group.add_argument("--manifest", type=Path,
                            help="JSONL manifest file (train.jsonl or dev.jsonl)")
    
    # Optional arguments
    parser.add_argument("--root_for_manifest", type=Path,
                        help="Root directory for resolving relative paths in manifest")
    parser.add_argument("--export_csv", type=str,
                        help="Export detailed statistics to CSV file")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress bar")
    
    args = parser.parse_args()
    
    print("🎵 FSR-2025 Hakka ASR Audio Duration Analyzer")
    print("=" * 50)
    
    start_time = time.time()
    
    # Get list of audio files
    if args.root:
        print(f"📁 Scanning audio files in: {args.root}")
        audio_files = scan_audio_files(args.root)
    else:
        print(f"📄 Loading from manifest: {args.manifest}")
        audio_files = load_from_manifest(args.manifest, args.root_for_manifest)
    
    if not audio_files:
        print("❌ No audio files found!")
        return 1
    
    print(f"🔍 Found {len(audio_files):,} audio files")
    
    # Analyze durations
    stats = analyze_durations(audio_files, show_progress=not args.quiet)
    
    # Print results
    print_statistics(stats)
    
    # Export CSV if requested
    if args.export_csv:
        export_to_csv(stats, args.export_csv)
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ Analysis completed in {elapsed:.1f} seconds")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error: {e}")
        sys.exit(1)