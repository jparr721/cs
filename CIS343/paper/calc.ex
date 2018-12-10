defmodule Calculator do
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

  def calc_helper(a, b, op, prev) do
    output = if op == "+" do
      add(a, b)
    else if op == "-" do
      substract(a, b)
    else if op == "*" || op == "x" do
      multiply(a, b)
    else if op == "/" do
      divide(a, b)

    IO.inspect output, label: "output: "
    prev.append(output)
    IO.inspect show_total(prev), label: "total: "

    run_calculator(prev)
  end

  def run_calculator(prev) do
    case IO.read(:stdio, :line) do
      :eof -> :ok
      {:error, reason} -> IO.puts "Error: #{reason}"
      data->
        if data == "exit" do
          System.stop(0)
        else
          vals = String.split(data, " ", trim: true)
          calc_helper(vals[0], vals[1], vals[2], prev)
  end
end

