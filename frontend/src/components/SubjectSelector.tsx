import { Subject } from '@/types/chat';
import { Atom, Calculator, FlaskConical, Leaf, Sparkles } from 'lucide-react';
import { cn } from '@/lib/utils';

interface SubjectSelectorProps {
  selected: Subject;
  onChange: (subject: Subject) => void;
}

const subjects: { id: Subject; label: string; icon: React.ReactNode; color: string }[] = [
  { id: 'physics', label: 'Physics', icon: <Atom className="h-4 w-4" />, color: 'physics' },
  { id: 'maths', label: 'Mathematics', icon: <Calculator className="h-4 w-4" />, color: 'maths' },
  { id: 'chemistry', label: 'Chemistry', icon: <FlaskConical className="h-4 w-4" />, color: 'chemistry' },
  { id: 'biology', label: 'Biology', icon: <Leaf className="h-4 w-4" />, color: 'biology' },
  { id: 'general', label: 'General', icon: <Sparkles className="h-4 w-4" />, color: 'primary' },
];

export function SubjectSelector({ selected, onChange }: SubjectSelectorProps) {
  return (
    <div className="flex flex-wrap gap-2">
      {subjects.map((subject) => (
        <button
          key={subject.id}
          onClick={() => onChange(subject.id)}
          className={cn(
            "flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium transition-all duration-300",
            selected === subject.id
              ? subject.color === 'physics'
                ? "bg-physics/20 text-physics border border-physics/30 shadow-[0_0_20px_hsl(199_89%_48%_/_0.2)]"
                : subject.color === 'maths'
                ? "bg-maths/20 text-maths border border-maths/30 shadow-[0_0_20px_hsl(280_65%_60%_/_0.2)]"
                : subject.color === 'chemistry'
                ? "bg-chemistry/20 text-chemistry border border-chemistry/30 shadow-[0_0_20px_hsl(142_71%_45%_/_0.2)]"
                : subject.color === 'biology'
                ? "bg-biology/20 text-biology border border-biology/30 shadow-[0_0_20px_hsl(35_91%_55%_/_0.2)]"
                : "bg-primary/20 text-primary border border-primary/30 shadow-[0_0_20px_hsl(174_72%_46%_/_0.2)]"
              : "bg-secondary/50 text-muted-foreground border border-transparent hover:bg-secondary hover:text-foreground"
          )}
        >
          {subject.icon}
          <span className="hidden sm:inline">{subject.label}</span>
        </button>
      ))}
    </div>
  );
}
