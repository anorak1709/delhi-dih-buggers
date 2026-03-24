import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export default function Tooltip({ text, children, position = 'top' }) {
  const [show, setShow] = useState(false);

  if (!text) return children;

  const isTop = position === 'top';

  return (
    <span
      className="relative inline-flex items-center gap-1 cursor-help"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      {children}
      <svg viewBox="0 0 16 16" fill="currentColor" className="w-3 h-3 text-surface-600 shrink-0">
        <path d="M8 1a7 7 0 100 14A7 7 0 008 1zm0 12.5a5.5 5.5 0 110-11 5.5 5.5 0 010 11zM8 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 018 5zm0 6.25a.75.75 0 100-1.5.75.75 0 000 1.5z" />
      </svg>
      <AnimatePresence>
        {show && (
          <motion.span
            initial={{ opacity: 0, y: isTop ? 4 : -4, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: isTop ? 4 : -4, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className={`
              absolute z-50 pointer-events-none px-3 py-2 text-xs leading-relaxed
              max-w-xs w-max rounded-lg shadow-lg
              backdrop-blur-md
              bg-[rgba(22,27,34,0.92)] dark:bg-[rgba(22,27,34,0.92)] bg-white/95
              border border-accent/15
              text-surface-300 dark:text-surface-300 text-surface-600
              ${isTop ? 'bottom-full mb-2 left-1/2 -translate-x-1/2' : 'top-full mt-2 left-1/2 -translate-x-1/2'}
            `.replace(/\n\s+/g, ' ').trim()}
          >
            {text}
          </motion.span>
        )}
      </AnimatePresence>
    </span>
  );
}
