export default function Input({ label, className = '', ...props }) {
  return (
    <div className={className}>
      {label && (
        <label className="block text-xs font-medium text-surface-400 dark:text-surface-400 text-surface-500 mb-1.5 uppercase tracking-wider">
          {label}
        </label>
      )}
      <input
        className={`
          w-full rounded-lg px-3.5 py-2 text-sm
          bg-surface-850/80 dark:bg-surface-850/80 bg-surface-50
          border border-surface-700/60 dark:border-surface-700/60 border-surface-200
          text-surface-50 dark:text-surface-50 text-surface-900
          placeholder:text-surface-600 dark:placeholder:text-surface-600 placeholder:text-surface-400
          focus:outline-none focus:border-accent/50 focus:ring-1 focus:ring-accent/20
          transition-colors duration-200
        `.replace(/\n\s+/g, ' ').trim()}
        {...props}
      />
    </div>
  );
}

export function Select({ label, children, className = '', ...props }) {
  return (
    <div className={className}>
      {label && (
        <label className="block text-xs font-medium text-surface-400 dark:text-surface-400 text-surface-500 mb-1.5 uppercase tracking-wider">
          {label}
        </label>
      )}
      <select
        className={`
          w-full rounded-lg px-3.5 py-2 text-sm
          bg-surface-850/80 dark:bg-surface-850/80 bg-surface-50
          border border-surface-700/60 dark:border-surface-700/60 border-surface-200
          text-surface-50 dark:text-surface-50 text-surface-900
          focus:outline-none focus:border-accent/50 focus:ring-1 focus:ring-accent/20
          transition-colors duration-200 cursor-pointer
        `.replace(/\n\s+/g, ' ').trim()}
        {...props}
      >
        {children}
      </select>
    </div>
  );
}
