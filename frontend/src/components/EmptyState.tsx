import { Atom, Calculator, FlaskConical, Leaf } from 'lucide-react';

const suggestions = [
  {
    icon: <Atom className="h-5 w-5" />,
    title: "Physics",
    example: "Explain the concept of quantum entanglement",
    color: "physics",
  },
  {
    icon: <Calculator className="h-5 w-5" />,
    title: "Mathematics",
    example: "Solve this integral: ∫x²dx",
    color: "maths",
  },
  {
    icon: <FlaskConical className="h-5 w-5" />,
    title: "Chemistry",
    example: "What happens in a redox reaction?",
    color: "chemistry",
  },
  {
    icon: <Leaf className="h-5 w-5" />,
    title: "Biology",
    example: "Describe the process of photosynthesis",
    color: "biology",
  },
];

interface EmptyStateProps {
  onSuggestionClick: (example: string) => void;
}

export function EmptyState({ onSuggestionClick }: EmptyStateProps) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center px-4 py-12">
      <div className="mb-8 text-center">
        <h2 className="font-display text-2xl font-semibold text-gradient mb-2">
          Academic AI Assistant
        </h2>
        <p className="text-muted-foreground max-w-md">
          Ask questions, upload images of problems, and get detailed explanations
          powered by Gemma3n running locally.
        </p>
      </div>

      <div className="grid w-full max-w-2xl gap-3 sm:grid-cols-2">
        {suggestions.map((suggestion) => (
          <button
            key={suggestion.title}
            onClick={() => onSuggestionClick(suggestion.example)}
            className={`group flex flex-col items-start gap-3 rounded-xl border border-border/50 bg-card/30 p-4 text-left transition-all duration-300 hover:border-${suggestion.color}/30 hover:bg-card/60 hover:shadow-lg`}
          >
            <div
              className={`flex h-10 w-10 items-center justify-center rounded-lg transition-colors ${
                suggestion.color === 'physics'
                  ? 'bg-physics/10 text-physics group-hover:bg-physics/20'
                  : suggestion.color === 'maths'
                  ? 'bg-maths/10 text-maths group-hover:bg-maths/20'
                  : suggestion.color === 'chemistry'
                  ? 'bg-chemistry/10 text-chemistry group-hover:bg-chemistry/20'
                  : 'bg-biology/10 text-biology group-hover:bg-biology/20'
              }`}
            >
              {suggestion.icon}
            </div>
            <div>
              <h3 className="font-medium text-foreground">{suggestion.title}</h3>
              <p className="mt-1 text-sm text-muted-foreground line-clamp-2">
                "{suggestion.example}"
              </p>
            </div>
          </button>
        ))}
      </div>

      <div className="mt-8 flex items-center gap-2 text-xs text-muted-foreground/60">
        <span className="h-2 w-2 rounded-full bg-primary animate-pulse" />
        <span>Powered by Gemma3n • Runs locally for privacy</span>
      </div>
    </div>
  );
}
