import { motion } from 'framer-motion';

export default function Loading({ text = 'Loading...' }) {
  return (
    <div className="flex flex-col items-center justify-center py-12 gap-4">
      <div className="relative w-14 h-14">
        {/* Outer ring */}
        <motion.div
          className="absolute inset-0 rounded-full border-2 border-surface-700/30"
          animate={{ rotate: 360 }}
          transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
        >
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-2 h-2 bg-accent rounded-full shadow-lg shadow-accent/40" />
        </motion.div>
        {/* Middle ring */}
        <motion.div
          className="absolute inset-2 rounded-full border-2 border-surface-700/20"
          animate={{ rotate: -360 }}
          transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
        >
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-1.5 h-1.5 bg-accent/60 rounded-full" />
        </motion.div>
        {/* Inner glow */}
        <motion.div
          className="absolute inset-4 rounded-full bg-accent/10"
          animate={{ scale: [1, 1.3, 1], opacity: [0.3, 0.6, 0.3] }}
          transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
        />
      </div>
      <motion.p
        className="text-xs text-surface-500 tracking-wide"
        animate={{ opacity: [0.5, 1, 0.5] }}
        transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
      >
        {text}
      </motion.p>
    </div>
  );
}

export function Skeleton({ className = '' }) {
  return (
    <div
      className={`animate-pulse rounded-lg bg-surface-700/40 dark:bg-surface-700/40 bg-surface-200 ${className}`}
    />
  );
}
