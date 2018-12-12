defmodule Calc do
  @moduledoc """
  Documentation for Calc.
  """
  def add(a, b) do
    a + b
  end

  def subtract(a, b) do
    a - b
  end

  def multiply(a, b) do
    a * b
  end

  def divide(a, b) do
    cond do
      b == 0 -> 0
      b != 0 -> a / b
    end
  end

  def show_total(lst) do
    Enum.sum(lst)
  end

  def calculate(line, prev) do
    if line[0] == "quit" do
      show_total(prev)
    else
      output = 0
      output =
        case line[3] do
          {:ok, "+"} -> add(line[0], line[1])
          {:ok, "-"} -> subtract(line[0], line[1])
          {:ok, "*"} -> multiply(line[0], line[1])
          {:ok, "/"} -> divide(line[0], line[1])
        end
      Enum.concat(prev, output)
    end
  end

  def parse(line) do
    vals = String.split(line, " ", trim: true)
    vals
  end

  def sub_main(prev) do
    input = IO.gets("> ")
    input
    |> parse
    |> calculate(prev)
    |> IO.puts
    sub_main(prev)
  end


  def main() do
    prev = []
    sub_main(prev)
  end

end
