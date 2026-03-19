import { motion } from 'framer-motion';

export default function Card({ children, className = '', hover = true, delay = 0, ...props }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16, filter: 'blur(4px)' }}
      animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
      transition={{
        type: 'spring',
        stiffness: 260,
        damping: 22,
        delay,
      }}
      whileHover={hover ? { y: -2, transition: { duration: 0.2 } } : undefined}
      className={`
        rounded-xl border border-surface-700/80
        bg-surface-800/60 backdrop-blur-sm
        dark:bg-surface-800/60 dark:border-surface-700/80
        bg-white border-surface-200
        ${hover ? 'card-glow' : ''}
        p-5 transition-shadow duration-300
        ${className}
      `.replace(/\n\s+/g, ' ').trim()}
      {...props}
    >
      {children}
    </motion.div>
  );
}

export function CardHeader({ title, subtitle, action, className = '' }) {
  return (
    <div className={`flex items-center justify-between mb-4 ${className}`}>
      <div>
        <h3 className="text-sm font-semibold uppercase tracking-wider text-surface-400 dark:text-surface-400 text-surface-500">
          {title}
        </h3>
        {subtitle && (
          <p className="text-xs text-surface-500 dark:text-surface-500 text-surface-400 mt-0.5">{subtitle}</p>
        )}
      </div>
      {action && <div>{action}</div>}
    </div>
  );
}
