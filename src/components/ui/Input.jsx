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
          w-full rounded-xl px-4 py-2.5 text-sm
          glass-input
          text-surface-50 dark:text-surface-50 text-surface-900
          placeholder:text-surface-600 dark:placeholder:text-surface-600 placeholder:text-surface-400
          focus:outline-none focus:border-accent/40 focus:ring-2 focus:ring-accent/15
          focus:shadow-[0_0_20px_rgba(201,152,90,0.08)]
          transition-all duration-300
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
          w-full rounded-xl px-4 py-2.5 text-sm
          glass-input
          text-surface-50 dark:text-surface-50 text-surface-900
          focus:outline-none focus:border-accent/40 focus:ring-2 focus:ring-accent/15
          focus:shadow-[0_0_20px_rgba(201,152,90,0.08)]
          transition-all duration-300 cursor-pointer
        `.replace(/\n\s+/g, ' ').trim()}
        {...props}
      >
        {children}
      </select>
    </div>
  );
}
