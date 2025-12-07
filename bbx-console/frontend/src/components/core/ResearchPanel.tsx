
import { useState, useRef, useEffect } from 'react'
import { search, browse, type SearchResponse, type SearchResult } from '@/services/api'
import ReactMarkdown from 'react-markdown'
import { useUIStore } from '@/stores/uiStore'

export function ResearchPanel() {
    const [query, setQuery] = useState('')
    const [isSearching, setIsSearching] = useState(false)
    const [results, setResults] = useState<SearchResponse | null>(null)
    const [activeUrl, setActiveUrl] = useState<string | null>(null)
    const [browsedContent, setBrowsedContent] = useState<string | null>(null)
    const [isBrowsing, setIsBrowsing] = useState(false)
    const inputRef = useRef<HTMLInputElement>(null)

    useEffect(() => {
        inputRef.current?.focus()
    }, [])

    const handleSearch = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!query.trim()) return

        setIsSearching(true)
        setResults(null)
        setActiveUrl(null)
        setBrowsedContent(null)

        try {
            const data = await search(query)
            setResults(data)
        } catch (err) {
            console.error(err)
        } finally {
            setIsSearching(false)
        }
    }

    const handleBrowse = async (result: SearchResult) => {
        setActiveUrl(result.url)
        setIsBrowsing(true)
        // Optimistic UI - show snippet first
        setBrowsedContent(`### Reading: ${result.title}\n\n*Visiting ${result.url}...*\n\n> ${result.content}`)

        try {
            const data = await browse(result.url)
            setBrowsedContent(`## ${result.title}\nSource: [${result.url}](${result.url})\n\n---\n\n${data.text}`)
        } catch (err) {
            setBrowsedContent(`### Error reading content\n\nFailed to browse ${result.url}`)
        } finally {
            setIsBrowsing(false)
        }
    }

    return (
        <div className="flex-1 flex flex-col h-full overflow-hidden bg-[var(--bg-primary)] text-[var(--text-primary)]">
            {/* Top Bar - Search */}
            <div className="flex-none p-6 border-b border-[var(--border-color)] glass">
                <form onSubmit={handleSearch} className="max-w-4xl mx-auto flex gap-4">
                    <input
                        ref={inputRef}
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Ask anything (e.g. 'How to solve error 137 in Docker')..."
                        className="flex-1 bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-xl px-4 py-3 focus:outline-none focus:ring-1 focus:ring-[var(--accent-primary)] text-lg"
                    />
                    <button
                        type="submit"
                        disabled={isSearching}
                        className="bg-[var(--accent-primary)] text-white px-6 py-3 rounded-xl hover:opacity-90 disabled:opacity-50 font-medium transition-colors"
                    >
                        {isSearching ? 'Searching...' : 'Search'}
                    </button>
                </form>
            </div>

            {/* Main Content */}
            <div className="flex-1 flex overflow-hidden">
                {/* Left: Results List (if results exist) */}
                {results && (
                    <div className={`${activeUrl ? 'w-1/3' : 'w-full max-w-4xl mx-auto'} flex-none overflow-y-auto p-6 border-r border-[var(--border-color)] transition-all duration-300`}>
                        <h2 className="text-sm font-bold uppercase tracking-wider text-[var(--text-muted)] mb-4">Sources</h2>

                        <div className="space-y-4">
                            {results.results.map((res, i) => (
                                <div
                                    key={i}
                                    onClick={() => handleBrowse(res)}
                                    className={`p-4 rounded-xl border cursor-pointer transition-all hover:border-[var(--accent-primary)] hover:bg-[var(--bg-secondary)] ${activeUrl === res.url ? 'border-[var(--accent-primary)] bg-[var(--bg-secondary)] ring-1 ring-[var(--accent-primary)]' : 'border-[var(--border-color)] glass'}`}
                                >
                                    <div className="flex items-center gap-2 mb-2">
                                        <div className="w-4 h-4 rounded-full bg-blue-500/20 text-blue-500 flex items-center justify-center text-[10px] font-bold">
                                            {i + 1}
                                        </div>
                                        <span className="text-xs text-[var(--text-muted)] truncate max-w-[200px]">{new URL(res.url).hostname}</span>
                                    </div>
                                    <h3 className="font-semibold mb-1 line-clamp-2">{res.title}</h3>
                                    <p className="text-sm text-[var(--text-muted)] line-clamp-3 leading-relaxed">{res.content}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Right: Content Reader (if active) */}
                {activeUrl && (
                    <div className="flex-1 overflow-y-auto p-8 bg-[var(--bg-secondary)]">
                        <div className="max-w-3xl mx-auto prose prose-invert">
                            {isBrowsing && !browsedContent && (
                                <div className="flex items-center justify-center py-20">
                                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[var(--accent-primary)]"></div>
                                </div>
                            )}
                            {browsedContent && (
                                <ReactMarkdown>{browsedContent}</ReactMarkdown>
                            )}
                        </div>
                    </div>
                )}

                {/* Empty State */}
                {!results && !isSearching && (
                    <div className="flex-1 flex items-center justify-center text-[var(--text-muted)]">
                        <div className="text-center">
                            <div className="text-6xl mb-4 opacity-20">🔍</div>
                            <p>Type above to start your sovereign research.</p>
                            <p className="text-sm mt-2 opacity-50">Local implementation using SearXNG & Headless Chrome</p>
                        </div>
                    </div>
                )}

                {/* Loading State */}
                {isSearching && !results && (
                    <div className="flex-1 flex items-center justify-center">
                        <div className="flex flex-col items-center gap-4">
                            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-[var(--accent-primary)]"></div>
                            <p className="text-[var(--text-muted)] animate-pulse">Searching the deep web...</p>
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}
