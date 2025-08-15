-- Schedule the experiment runner to check for queued experiments every minute
SELECT cron.schedule(
    'process-experiments',
    '* * * * *', -- Every minute
    $$
    SELECT net.http_post(
        url := 'https://ktgzaciforpkhnfdtbqm.supabase.co/functions/v1/experiment-runner',
        headers := '{"Content-Type": "application/json", "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt0Z3phY2lmb3Jwa2huZmR0YnFtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDUxMzk2NjYsImV4cCI6MjA2MDcxNTY2Nn0.o0uZDsoRWNTLHL2qOxWSNI-rHDSMW7ixPc1aJwYo28k"}'::jsonb,
        body := '{}'::jsonb
    );
    $$
);