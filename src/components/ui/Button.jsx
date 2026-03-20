import { motion, AnimatePresence } from 'framer-motion';

const variants = {
  primary: `
    bg-gradient-to-r from-accent to-accent-light text-surface-900 font-semibold
    shadow-lg shadow-accent/20
    hover:shadow-xl hover:shadow-accent/30
    active:shadow-md
  `,
  secondary: `
    glass-input text-surface-200 dark:text-surface-200 text-surface-700
    hover:border-accent/30
  `,
  ghost: `
    text-surface-400 hover:text-surface-200 dark:hover:text-surface-200 hover:text-surface-700
    hover:bg-surface-700/20 dark:hover:bg-surface-700/20 hover:bg-surface-100
  `,
  danger: `
    bg-down/10 text-down border border-down/20
    hover:bg-down/20 hover:border-down/30
    hover:shadow-lg hover:shadow-down/10
  `,
};

const sizes = {
  sm: 'px-3.5 py-1.5 text-xs rounded-lg',
  md: 'px-5 py-2.5 text-sm rounded-xl',
  lg: 'px-7 py-3 text-sm rounded-xl',
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
      whileHover={!disabled && !loading ? { scale: 1.04, y: -1 } : undefined}
      whileTap={!disabled && !loading ? { scale: 0.96 } : undefined}
      transition={{ type: 'spring', stiffness: 500, damping: 18 }}
      disabled={loading || disabled}
      className={`
        inline-flex items-center justify-center gap-2 font-medium
        transition-all duration-300 cursor-pointer
        disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none
        ${variants[variant]} ${sizes[size]} ${className}
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
            className="animate-spin h-4 w-4"
            viewBox="0 0 24 24"
            fill="none"
          >
            <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" className="opacity-20" />
            <path d="M12 2a10 10 0 019.95 9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
          </motion.svg>
        )}
      </AnimatePresence>
      {children}
    </motion.button>
  );
}
