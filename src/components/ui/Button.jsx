import { motion, AnimatePresence } from 'framer-motion';

const styleVariants = {
  primary:
    'bg-accent text-surface-900 hover:bg-accent-light font-semibold shadow-sm shadow-accent/10',
  secondary:
    'bg-surface-700/50 dark:bg-surface-700/50 bg-surface-100 text-surface-300 dark:text-surface-300 text-surface-600 hover:bg-surface-700 dark:hover:bg-surface-700 hover:bg-surface-200 border border-surface-700/60 dark:border-surface-700/60 border-surface-200',
  ghost:
    'text-surface-400 hover:text-surface-50 dark:hover:text-surface-50 hover:text-surface-900 hover:bg-surface-700/30 dark:hover:bg-surface-700/30 hover:bg-surface-100',
  danger:
    'bg-down/10 text-down hover:bg-down/20 border border-down/20',
};

const sizes = {
  sm: 'px-3 py-1.5 text-xs',
  md: 'px-4 py-2 text-sm',
  lg: 'px-6 py-2.5 text-sm',
};

export default function Button({
  children,
  variant = 'primary',
  size = 'md',
  loading = false,
  disabled = false,
  className = '',
  ...props
}) {
  return (
    <motion.button
      whileHover={!disabled && !loading ? { scale: 1.03 } : undefined}
      whileTap={!disabled && !loading ? { scale: 0.96 } : undefined}
      transition={{ type: 'spring', stiffness: 400, damping: 17 }}
      disabled={disabled || loading}
      className={`
        inline-flex items-center justify-center gap-2
        rounded-lg font-medium
        transition-colors duration-200
        disabled:opacity-40 disabled:cursor-not-allowed
        cursor-pointer select-none
        ${styleVariants[variant]}
        ${sizes[size]}
        ${className}
      `.replace(/\n\s+/g, ' ').trim()}
      {...props}
    >
      <AnimatePresence mode="wait">
        {loading && (
          <motion.svg
            key="spinner"
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.5 }}
            className="animate-spin h-3.5 w-3.5"
            viewBox="0 0 24 24"
            fill="none"
          >
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
          </motion.svg>
        )}
      </AnimatePresence>
      {children}
    </motion.button>
  );
}
