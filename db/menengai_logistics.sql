-- Menengai Orders Logistics: trips shown on the dashboard's Trips tab.
-- Run once in the Supabase SQL editor of the ControlTech HOS project.
-- Only the server (Supabase secret key) reads and writes this table: RLS is on with no policies,
-- and the browser-facing roles have no privileges.

create extension if not exists pgcrypto;

create table if not exists menengai_trips (
  id uuid primary key default gen_random_uuid(),
  resource_id bigint not null,
  -- Wialon Logistics route uid; one trip per route (uploads and routes read from Wialon history).
  source_key text not null,
  -- "upload" (dispatched from this app) or "wialon" (finished route read from Wialon Logistics history).
  source text not null,
  route_uid bigint,
  route_name text not null,
  unit_id bigint not null,
  unit_name text not null default '',
  warehouse text not null default '',
  day date not null,
  window_from timestamptz not null,
  window_to timestamptz not null,
  -- Planned schedule: depot, stops with ETAs, distance, geometry and visit rules.
  plan jsonb not null,
  files text not null default '',
  created_at timestamptz not null default now(),
  created_by text not null default '',
  -- Planned-vs-actual of a finished trip (without the path), so lists need not reload positions from Wialon.
  result jsonb,
  result_at timestamptz,
  -- A removed trip is kept so it is not read again from Wialon history.
  removed_at timestamptz
);
create unique index if not exists menengai_trips_source_key on menengai_trips (source_key);
create index if not exists menengai_trips_day on menengai_trips (resource_id, day desc);

alter table menengai_trips enable row level security;
revoke all on menengai_trips from anon, authenticated;
