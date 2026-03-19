import { motion } from 'framer-motion';

export default function Loading({ text = 'Loading...' }) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
      transition={{ duration: 0.3 }}
      className="flex flex-col items-center justify-center py-16 gap-4"
    >
      <div className="relative">
        <div className="w-10 h-10 rounded-full border-2 border-surface-700 dark:border-surface-700 border-surface-200" />
        <motion.div
          className="absolute inset-0 w-10 h-10 rounded-full border-2 border-transparent border-t-accent"
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
        />
        <motion.div
          className="absolute inset-1 w-8 h-8 rounded-full border-2 border-transparent border-b-accent/40"
          animate={{ rotate: -360 }}
          transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
        />
      </div>
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="text-xs text-surface-500 tracking-wide uppercase"
      >
        {text}
      </motion.p>
    </motion.div>
  );
}

export function Skeleton({ className = '' }) {
  return (
    <div
      className={`animate-pulse rounded-lg bg-surface-700/40 dark:bg-surface-700/40 bg-surface-200 ${className}`}
    />
  );
}
