#!/usr/bin/env python3
"""
ScholarQA Trace Display Tool

This script displays ScholarQA trace results in a beautiful, readable format.
Usage: python display.py <trace_file.json>
"""

import json
import sys
from datetime import datetime
from typing import Dict, Any, List
import argparse


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def format_timestamp(timestamp_str: str) -> str:
    """Format ISO timestamp to readable format"""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp_str


def format_cost(cost: float) -> str:
    """Format cost with currency symbol"""
    return f"${cost:.4f}"


def format_tokens(tokens: Dict[str, int]) -> str:
    """Format token usage information"""
    total = tokens.get('total', 0)
    input_tokens = tokens.get('input', 0)
    output_tokens = tokens.get('output', 0)
    reasoning = tokens.get('reasoning', 0)

    result = f"Total: {total:,}"
    if input_tokens or output_tokens:
        result += f" (In: {input_tokens:,}, Out: {output_tokens:,}"
        if reasoning:
            result += f", Reasoning: {reasoning:,}"
        result += ")"
    return result


def print_header(title: str, color: str = Colors.HEADER):
    """Print a formatted header"""
    print(f"\n{color}{Colors.BOLD}{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}{Colors.END}\n")


def print_section(title: str, content: str = "", color: str = Colors.CYAN):
    """Print a formatted section"""
    print(f"{color}{Colors.BOLD}{title}{Colors.END}")
    if content:
        print(f"{content}\n")
    else:
        print()


def display_summary_info(data: Dict[str, Any]):
    """Display basic query and execution information"""
    print_header("Query Information", Colors.BLUE)

    query = data.get('query', 'N/A')
    task_id = data.get('task_id', 'N/A')
    timestamp = format_timestamp(data.get('timestamp', ''))
    user_id = data.get('user_id', 'N/A')
    total_cost = format_cost(data.get('total_cost', 0))

    print(f"{Colors.BOLD}Query:{Colors.END} {query}")
    print(f"{Colors.BOLD}Task ID:{Colors.END} {task_id}")
    print(f"{Colors.BOLD}User:{Colors.END} {user_id}")
    print(f"{Colors.BOLD}Timestamp:{Colors.END} {timestamp}")
    print(f"{Colors.BOLD}Total Cost:{Colors.END} {total_cost}")

    # Token usage
    if 'tokens' in data:
        tokens_str = format_tokens(data['tokens'])
        print(f"{Colors.BOLD}Token Usage:{Colors.END} {tokens_str}")


def display_retrieval_stats(data: Dict[str, Any]):
    """Display retrieval statistics"""
    print_header("Retrieval Statistics", Colors.GREEN)

    n_retrieval = data.get('n_retrieval', 0)
    n_retrieved = data.get('n_retrieved', 0)
    n_candidates = data.get('n_candidates', 0)
    n_rerank = data.get('n_rerank', 0)

    print(f"{Colors.BOLD}Requested:{Colors.END} {n_retrieval}")
    print(f"{Colors.BOLD}Retrieved:{Colors.END} {n_retrieved}")
    print(f"{Colors.BOLD}Candidates:{Colors.END} {n_candidates}")
    print(f"{Colors.BOLD}Rerank Limit:{Colors.END} {n_rerank if n_rerank > 0 else 'Disabled'}")

    # Decomposed query info
    if 'decomposed_query' in data:
        dq = data['decomposed_query']
        print(f"\n{Colors.BOLD}Query Rewriting:{Colors.END}")
        print(f"  Original: {data.get('query', 'N/A')}")
        print(f"  Rewritten: {dq.get('rewritten_query', 'N/A')}")
        print(f"  Keywords: {dq.get('keyword_query', 'N/A')}")

        if 'search_filters' in dq:
            filters = dq['search_filters']
            if filters:
                print(f"  Filters: {', '.join(f'{k}={v}' for k, v in filters.items())}")


def display_papers_summary(candidates: List[Dict[str, Any]]):
    """Display summary of retrieved papers"""
    print_header("Retrieved Papers Summary", Colors.YELLOW)

    print(f"{Colors.BOLD}Total Papers:{Colors.END} {len(candidates)}")

    # Count by venue
    venues = {}
    years = {}
    for paper in candidates:
        venue = paper.get('venue', 'Unknown')
        year = paper.get('year', 'Unknown')
        venues[venue] = venues.get(venue, 0) + 1
        years[year] = years.get(year, 0) + 1

    print(f"\n{Colors.BOLD}By Venue:{Colors.END}")
    for venue, count in sorted(venues.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {venue}: {count}")

    print(f"\n{Colors.BOLD}By Year:{Colors.END}")
    for year, count in sorted(years.items(), key=lambda x: x[0], reverse=True)[:5]:
        print(f"  {year}: {count}")


def display_sample_papers(candidates: List[Dict[str, Any]], limit: int = 3):
    """Display sample papers with their content"""
    print_header("Sample Retrieved Papers", Colors.CYAN)

    for i, paper in enumerate(candidates[:limit]):
        print(f"{Colors.BOLD}Paper {i+1}:{Colors.END}")
        print(f"  {Colors.BOLD}ID:{Colors.END} {paper.get('corpus_id', 'N/A')}")
        print(f"  {Colors.BOLD}Title:{Colors.END} {paper.get('title', 'N/A')[:100]}...")
        print(f"  {Colors.BOLD}Venue:{Colors.END} {paper.get('venue', 'N/A')}")
        print(f"  {Colors.BOLD}Year:{Colors.END} {paper.get('year', 'N/A')}")
        print(f"  {Colors.BOLD}Citations:{Colors.END} {paper.get('citation_count', 0)}")

        # Show sample sentences
        sentences = paper.get('sentences', [])
        if sentences:
            print(f"  {Colors.BOLD}Sample Text:{Colors.END}")
            sample_text = sentences[0].get('text', '')[:200]
            print(f"    {sample_text}...")
        print()


def display_final_results(data: Dict[str, Any]):
    """Display the final generated answer"""
    if 'summary' not in data:
        print_section("No final summary found in trace data", color=Colors.RED)
        return

    summary = data['summary']
    sections = summary.get('sections', [])

    print_header("Generated Answer", Colors.GREEN)

    # Summary stats
    if 'cost' in summary:
        print(f"{Colors.BOLD}Generation Cost:{Colors.END} {format_cost(summary['cost'])}")
    if 'tokens' in summary:
        print(f"{Colors.BOLD}Generation Tokens:{Colors.END} {format_tokens(summary['tokens'])}")

    print(f"\n{Colors.BOLD}Number of Sections:{Colors.END} {len(sections)}")

    # Display each section
    for i, section in enumerate(sections, 1):
        print(f"\n{Colors.BOLD}{Colors.UNDERLINE}Section {i}: {section.get('title', 'Untitled')}{Colors.END}")

        # TL;DR
        if section.get('tldr'):
            print(f"{Colors.BOLD}TL;DR:{Colors.END} {section['tldr']}")

        # Main text (truncated)
        text = section.get('text', '')
        if text:
            # Show first 300 characters
            preview = text[:].replace('\n', ' ').strip()
            print(f"{Colors.BOLD}Content:{Colors.END} {preview}")

        # Format and model info
        format_type = section.get('format', 'N/A')
        model = section.get('model', 'N/A')
        print(f"{Colors.BOLD}Format:{Colors.END} {format_type} | {Colors.BOLD}Model:{Colors.END} {model}")


def display_clustering_info(data: Dict[str, Any]):
    """Display clustering information if available"""
    if 'clustering' not in data:
        return

    clustering = data['clustering']
    if 'clusters' not in clustering:
        return

    print_header("Quote Clustering", Colors.YELLOW)

    clusters = clustering['clusters']
    print(f"{Colors.BOLD}Number of Clusters:{Colors.END} {len(clusters)}")

    for cluster_name, quote_ids in clusters.items():
        print(f"\n{Colors.BOLD}{cluster_name}:{Colors.END}")
        print(f"  {len(quote_ids)} quotes: {quote_ids[:10]}{'...' if len(quote_ids) > 10 else ''}")


def main():
    parser = argparse.ArgumentParser(description='Display ScholarQA trace results')
    parser.add_argument('trace_file', help='Path to the trace JSON file')
    parser.add_argument('--no-color', action='store_true', help='Disable colored output')
    parser.add_argument('--full-papers', action='store_true', help='Show all papers instead of just samples')
    parser.add_argument('--full-text', action='store_true', help='Show full section text instead of previews')

    args = parser.parse_args()

    # Disable colors if requested
    if args.no_color:
        for attr in dir(Colors):
            if not attr.startswith('_'):
                setattr(Colors, attr, '')

    try:
        with open(args.trace_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"{Colors.RED}Error: File '{args.trace_file}' not found{Colors.END}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"{Colors.RED}Error: Invalid JSON in '{args.trace_file}': {e}{Colors.END}")
        sys.exit(1)

    # Display sections
    display_summary_info(data)
    display_retrieval_stats(data)

    candidates = data.get('candidates', [])
    if candidates:
        display_papers_summary(candidates)
        if args.full_papers:
            display_sample_papers(candidates, limit=len(candidates))
        else:
            display_sample_papers(candidates, limit=3)

    display_clustering_info(data)
    display_final_results(data)

    print(f"\n{Colors.GREEN}{Colors.BOLD}Display complete!{Colors.END}")


if __name__ == "__main__":
    main()